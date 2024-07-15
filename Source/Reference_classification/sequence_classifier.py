import torch
from torch import nn
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
