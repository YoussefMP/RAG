from torch import nn
from torchcrf import CRF
from transformers import AutoModel
import torch.nn.functional as F


class RobertaCRF(nn.Module):
    def __init__(self, model_name, num_labels):
        super(RobertaCRF, self).__init__()
        self.num_labels = num_labels
        self.roberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.hidden2tag = nn.Linear(self.roberta.config.hidden_size, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        emissions = self.hidden2tag(sequence_output)

        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
            return loss
        else:
            return self.crf.decode(emissions, mask=attention_mask.byte())

    def predict_with_confidence(self, input_ids, attention_mask=None, device="cpu"):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        emissions = self.hidden2tag(sequence_output)

        predictions = self.crf.decode(emissions, mask=attention_mask.byte())
        max_length = max([len(p) for p in predictions])
        predictions = [p + [0] * (max_length - len(p)) for p in predictions]

        # Compute softmax probabilities
        probabilities = F.softmax(emissions, dim=-1)

        # Extract the confidence for the predicted label for each token
        confidences = []
        for i, preds in enumerate(predictions):
            token_confidences = []
            for j, pred in enumerate(preds):
                token_confidences.append(probabilities[i, j, pred].item())
            confidences.append(token_confidences)

        return predictions, confidences
