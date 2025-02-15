import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from Source.Utils.labels import TAG2ID, POSSIBLE_RELATIONS
from Utils.decorators import *
from torchcrf import CRF
from transformers import AutoModel
import torch.nn.functional as F
import random
import numpy as np


DEBUG = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RobertaCRF(nn.Module):
    def __init__(self, model_name, num_labels):
        super(RobertaCRF, self).__init__()
        self.num_labels = num_labels
        self.roberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.hidden2tag = nn.Linear(self.roberta.config.hidden_size, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask=None, labels=None, threshold=None):

        if threshold:
            return self.predict_with_confidence(input_ids, attention_mask, threshold)

        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        emissions = self.hidden2tag(sequence_output)

        if labels is not None:
            # loss = -self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
            loss = -self.crf(emissions, labels, mask=attention_mask.byte())
            return loss
        else:
            return self.crf.decode(emissions, mask=attention_mask.byte())

    def predict_with_confidence(self, input_ids, attention_mask, threshold):
        """
        This method is a forward pass, but it also returns the confidence scores for each label predicted.
        :param input_ids:
        :param attention_mask:
        :param threshold:
        :return:
        """

        with torch.no_grad():
            outputs = self.roberta(input_ids, attention_mask=attention_mask)
            sequence_output = self.dropout(outputs[0])
            emissions = self.hidden2tag(sequence_output)
            predictions = self.crf.decode(emissions, mask=attention_mask.byte())

            # Compute softmax probabilities
            probabilities = F.softmax(emissions, dim=-1)

            # Extract the confidence for the predicted label for each token
            confident_predictions = []
            for i, preds in enumerate(predictions):
                prediction = []
                for j, pred in enumerate(preds):
                    if probabilities[i, j, pred].item() > threshold:
                        prediction.append(pred)
                    elif threshold-0.04 < probabilities[i, j, pred].item() < threshold:
                        prediction.append(9)  # Replace with uncertainty tag
                    else:
                        prediction.append(0)  # Replace with a neutral tag or None
                confident_predictions.append(prediction)

        return confident_predictions


class RefDissassembler(RobertaCRF):

    def __init__(self, model_name, num_labels, num_relations):
        super(RefDissassembler, self).__init__(model_name, num_labels)
        self.relation_hidden = torch.nn.Linear(self.roberta.config.hidden_size * 2,
                                               self.roberta.config.hidden_size * 2)
        self.relation_hidden_second = torch.nn.Linear(self.roberta.config.hidden_size * 2,
                                                      self.roberta.config.hidden_size)
        self.relation_classifier_layer = torch.nn.Linear(self.roberta.config.hidden_size, num_relations)
        self.loss_fn = BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask=None, labels=None, relations=None, threshold=None, r_threshold=0.7):
        """
        :param input_ids:
        :param attention_mask:
        :param labels:
        :param relations:
        :param threshold:
        :param r_threshold:
        :return:
                    :outputs: Loss function for training and evaluation: Set (loss, crf_output, relation_logits)
                                loss is the sum of the crf-loss + BCEWithLogitsLoss for the relation layer
        """

        # TODO: evaluation and production pipeline

        if threshold:
            pass

        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        # The ReLU function is here only to test the performance of a slightly different encoder architecture
        # hidden_output = torch.nn.ReLU()(self.hidden2tag(sequence_output))
        hidden_output = self.hidden2tag(sequence_output)
        decoded_crf = self.crf.decode(hidden_output, mask=attention_mask.byte())

        relation_logits = None
        rel_predictions = None
        golden_truth = None
        if (labels is None and len(decoded_crf) > 0) or len(labels) > 0:
            span_pairs = self.pair_sequence_embeddings(sequence_output, decoded_crf, labels)
            golden_truth, pairs_embeddings = self.prepare_golden_truth(span_pairs, relations)
            try:
                # relation_hidden = torch.nn.ReLU()(self.relation_hidden(pairs_embeddings))
                relation_hidden = self.relation_hidden(pairs_embeddings)
                relation_hidden = F.tanh(self.relation_hidden_second(relation_hidden))
                # relation_hidden = self.relation_hidden(pairs_embeddings)
            except TypeError as err:
                if pairs_embeddings is None:
                    return None, None
                else:
                    print(err)
                    print(f"pairs_embedding type = {type(pairs_embeddings)}")
                    print(f"relation_hidden output type = {type(self.relation_hidden(pairs_embeddings))}")

            relation_logits = self.relation_classifier_layer(relation_hidden)
            rel_predictions_probs = torch.sigmoid(relation_logits)
            rel_predictions = (rel_predictions_probs > r_threshold).int()

        if relations is not None:
            relation_loss = self.loss_fn(relation_logits.reshape(relation_logits.size(0)),
                                         torch.tensor(golden_truth, dtype=torch.float).to("cuda"))
            return -self.crf(hidden_output, labels, mask=attention_mask.byte()), relation_loss
        else:
            return decoded_crf, (golden_truth, rel_predictions)

    def pair_sequence_embeddings(self, sequence_output, decoded_crf, labels=None):
        """
        Pairs the embeddings of the sequences of tokens that are related to each other
        :param sequence_output: Embedding of the text [batch_size, seq_length, models_output_size]
        :param decoded_crf:     A list of labels for each token in the sequence [seq_length]
        :param labels:         A list of labels for each token in the sequence [seq_length]
        :return:
        """
        pairs_batches = []
        labels_to_use = labels.tolist() if labels is not None else decoded_crf

        # This loop iterates the predicted labels or the golden truth of the batch
        for eid, predicted_labels in enumerate(labels_to_use):
            # This logic takes a list of labels of tokens and saves the order at which the sequences appear
            # in labels_in_oder as well as their end index in borders
            # example: labels [0, 0, 0, 4, 4, 5, 5, 0, 0, 5, 5]
            #        : labels_in_order [0, 4, 5, 0, 5]
            #        : borders [3, 5, 7, 9]
            borders = []
            list_predicted_labels = predicted_labels
            labels_in_order = [list_predicted_labels[0]]
            old_label = list_predicted_labels[0]
            for pid, predicted_label in enumerate(list_predicted_labels):
                if old_label != predicted_label:
                    labels_in_order.append(predicted_label)
                    borders.append(pid)
                    old_label = predicted_label

                if sum(list_predicted_labels[pid:]) == 0:
                    break

            # This concatenates the embeddings of the tokens with the same label belonging to the same sequence
            # sequence_embeddings = list [ tuple (int, tensor[ float ]) ] // tensor.shape = [hidden_size, ]
            sequence_embeddings = []
            for i, b in enumerate(borders):
                current_label = labels_in_order[i]
                if current_label == 0:
                    continue
                if i == len(borders) - 1:
                    sequence_embeddings.append((current_label, sequence_output[eid][borders[i-1]:b].mean(dim=0)))
                    if labels_in_order[i+1] != 0:
                        sequence_embeddings.append((labels_in_order[i+1], sequence_output[eid][b:].mean(dim=0)))
                elif i == 0:
                    sequence_embeddings.append((current_label, sequence_output[eid][:b].mean(dim=0)))
                else:
                    sequence_embeddings.append((current_label, sequence_output[eid][borders[i-1]:b].mean(dim=0)))

            pairs_batches.append(self.construct_relation_matrix(sequence_embeddings))
        return pairs_batches

    @staticmethod
    def construct_relation_matrix(sequence_embeddings):
        """"""
        # TODO: introduce new ids counting the zero labeled sequences as well
        # TODO: set the entity to book relation
        # TODO : Idea was to pass len(borders) with the embeddings
        # TODO :
        relation_matrix = []
        twos_and_threes = []    # These are sequences that represent books and regulations
        fours_and_fives = []    # these are sequences that represent Articles and Sections
        if len(sequence_embeddings) > 1:
            for eid, (label, entity) in enumerate(reversed(sequence_embeddings)):
                if label == 2 or label == 3:
                    twos_and_threes.append((eid, label, entity))
                    continue
                elif label == 4 or label == 5:
                    fours_and_fives.append((eid, label, entity))

                jid = eid + 1
                for jlabel, jentity in reversed(sequence_embeddings[:-jid]):
                    if jlabel < label:
                        if jlabel in POSSIBLE_RELATIONS[label]:
                            pair_embedding = torch.concat((entity, jentity), dim=0).unsqueeze(0)
                            relation_matrix.append((
                                [eid, jid],
                                pair_embedding
                            ))
                    jid += 1

            if len(twos_and_threes) > 0:
                for fof in fours_and_fives:
                    for tot in twos_and_threes:
                        pair_embedding = torch.concat((fof[2], tot[2]), dim=0).unsqueeze(0)
                        relation_matrix.append((
                            [fof[0], tot[0]],
                            pair_embedding
                        ))

        return relation_matrix

    @staticmethod
    def prepare_golden_truth(span_pairs, relations):
        """
        :param span_pairs:
        :param relations:
        :return:
        """
        # Extract the embeddings for each pair and group them together into batches for efficiency
        pairs = []
        pairs_embeddings = []
        for sub_batch in span_pairs:
            sub_batch_pairs = []
            sub_batch_embeddings = []
            for pair, embedding in sub_batch:
                sub_batch_pairs.append(pair)
                sub_batch_embeddings.append(embedding)

            pairs.append(sub_batch_pairs)
            pairs_embeddings.append(torch.stack(sub_batch_embeddings) if sub_batch else torch.tensor([]))

        if relations is not None:
            # Convert the golden truth into a list of binary values (1 for matching pairs, 0 for non-matching pairs)
            golden_truth = []
            for pbid, pairs_batch in enumerate(pairs):
                for pair in pairs_batch:
                    if pair in relations[pbid].tolist():
                        golden_truth.append(1)
                    else:
                        golden_truth.append(0)
        else:
            golden_truth = pairs

        # Flatten pairs_embeddings for further processing
        try:
            pairs_embeddings = torch.cat([batch for batch in pairs_embeddings if batch.numel() > 0], dim=0)
        except RuntimeError as err:
            print(err)
            return None, None
        except Exception:
            print("Error occurred while flattening pairs_embeddings.")
        return golden_truth, pairs_embeddings

    ###############
    # Old Methods #
    ###############

    @deprecated
    def get_spans_pairs(self, sequence_output, decoded_crf, labels=None):
        """
        based on the sequence_output and the decoded_crf this function groups the embeddings of the tokens of each span
        :param sequence_output: Embedding of the text [batch_size, seq_length, models_output_size]
        :param decoded_crf:     A list of labels for each token in the sequence [seq_length]
        :param labels:         A list of labels for each token in the sequence [seq_length]
        :return:
        """

        # initialize empty tensor to store the results

        pairs_batch = []
        labels_to_use = labels.tolist() if labels is not None else decoded_crf

        # This loop iterates the predicted labels or the golden truth of the batch
        for eid, predicted_labels in enumerate(labels_to_use):

            # This logic takes a list of labels of tokens and saves the order at which they appear in labels_in_oder
            # as well as the borders; the indices of the end of each sequence
            # example: labels [0, 0, 0, 4, 4, 5, 5, 0, 0, 5, 5]
            #        : labels_in_order [0, 4, 5, 0, 5]
            borders = []
            list_predicted_labels = predicted_labels
            labels_in_order = [list_predicted_labels[0]]
            old_label = list_predicted_labels[0]
            for pid, predicted_label in enumerate(list_predicted_labels):
                if old_label != predicted_label:
                    labels_in_order.append(predicted_label)
                    borders.append(pid)
                    old_label = predicted_label

            # This concatenates the embeddings of the tokens with the same label belonging to the same sequence
            # spans_embeddings = list [ tuple (int, tensor[ float ]) ] // tensor.shape = [hidden_size, ]
            spans_embeddings = []
            for i, b in enumerate(borders):
                current_label = labels_in_order[i]
                if current_label == 0:
                    continue
                if i == len(borders) - 1:
                    spans_embeddings.append((current_label, sequence_output[eid][borders[i-1]:b].mean(dim=0)))
                    if labels_in_order[i+1] != 0:
                        spans_embeddings.append((labels_in_order[i+1], sequence_output[eid][b:].mean(dim=0)))
                elif i == 0:
                    spans_embeddings.append((current_label, sequence_output[eid][:b].mean(dim=0)))
                else:
                    spans_embeddings.append((current_label, sequence_output[eid][borders[i-1]:b].mean(dim=0)))

            spans_embeddings.sort(key=lambda x: x[0], reverse=True)
            pairs_batch.append(self.generate_relations_from_entities(spans_embeddings))

        # flatten the batch pairs into a one dimensional array
        flat_pairs_batch = [pair for example_pairs in pairs_batch for pair in example_pairs]
        pairs_batch = torch.cat(flat_pairs_batch, dim=0)

        return pairs_batch

    @staticmethod
    @deprecated
    def generate_relations_from_entities(entities):
        """
        This gets the entities in descending order of their labels and pairs all labels that can be paired together.
        :param entities:
        :return:
        """
        relations_set = []

        if len(entities) > 1:
            for start_entity in entities:
                if start_entity[0] == 2 or start_entity[0] == 3:
                    continue

                for end_entity in entities:
                    if end_entity[0] < start_entity[0]:
                        if end_entity[0] not in POSSIBLE_RELATIONS[start_entity[0]]:
                            continue
                        # relations_set.append((entities.index(start_entity), entities.index(end_entity)))
                        pair_embedding = torch.concat((start_entity[1], end_entity[1]), dim=0).unsqueeze(0)
                        relations_set.append(pair_embedding)
        return relations_set


