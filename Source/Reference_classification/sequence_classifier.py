import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from Utils.labels import TAG2ID, POSSIBLE_RELATIONS
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
            loss = -self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
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
                                               self.roberta.config.hidden_size)
        self.relation_classifier_layer = torch.nn.Linear(self.roberta.config.hidden_size, num_relations)
        self.loss_fn = BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask=None, labels=None, relations=None, threshold=None):
        """
        :param input_ids:
        :param attention_mask:
        :param labels:
        :param relations:
        :param threshold:
        :return:
                    :outputs: Loss function for training and evaluation: Set (loss, crf_output, relation_logits)
                                loss is the sum of the crf-loss + BCEWithLogitsLoss for the relation layer
        """

        if threshold:
            pass

        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        # The ReLU function is here only to test the performance of a slightly different encoder architecture
        hidden_output = torch.nn.ReLU()(self.hidden2tag(sequence_output))
        decoded_crf = self.crf.decode(hidden_output, mask=attention_mask.byte())

        span_pairs = self.get_spans_pairs(sequence_output, decoded_crf, labels)
        relation_hidden = torch.nn.ReLU()(self.relation_hidden(span_pairs))
        relation_logits = self.relation_classifier_layer(relation_hidden)

        if labels is not None:
            # Removed the reduction parameter from the 'crf'-call for more granular token level fine-tuning
            loss = (-self.crf(hidden_output, labels, mask=attention_mask.byte()) +
                    self.loss_fn(relation_logits,
                                 torch.tensor(relations, dtype=torch.float).unsqueeze(dim=1).to("cuda"))
                    )
            return loss

        return decoded_crf, relation_logits

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
        labels_to_use = labels if labels is not None else decoded_crf

        # This loop iterates the predicted labels or the golden truth of the batch
        for eid, predicted_labels in enumerate(labels_to_use):

            # this loop set the indices of the label spans
            borders = []

            list_predicted_labels = predicted_labels.tolist()
            labels_in_order = [list_predicted_labels[0]]
            old_label = list_predicted_labels[0]
            for pid, predicted_label in enumerate(list_predicted_labels):
                if old_label != predicted_label:
                    labels_in_order.append(predicted_label)
                    borders.append(pid)
                    old_label = predicted_label

            # This returns each span embeddings
            # spans_embeddings = list [ tuple (int, tensor[ float ]) ] tensor.shape = [span_length, hidden_size]
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
    def generate_relations_from_entities(entities):
        """
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




















