import torch
from torch import nn, optim
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig

class DistilBertForTFQA(DistilBertModel):
    """BERT model for QA and classification tasks.

    Parameters
    ----------
    config : transformers.BertConfig. Configuration class for BERT.

    Returns
    -------
    start_logits : torch.Tensor with shape (batch_size, sequence_size).
        Starting scores of each tokens.
    end_logits : torch.Tensor with shape (batch_size, sequence_size).
        Ending scores of each tokens.
    classifier_logits : torch.Tensor with shape (batch_size, num_classes).
        Classification scores of each labels.
    """

    def __init__(self, config, frozen_layers=0, hidden_layers=6, batch_size=16, max_seq_len=384):
        super(DistilBertForTFQA, self).__init__(config)
        self.bert = DistilBertModel(config)
        ct = 0
        for child in self.bert.children():
            ct +=1
            if ct < frozen_layers:
                child.requires_grad = False
        self.qa_outputs = nn.Linear(config.hidden_size, 2)  # start/end
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(config.dropout) #RekhaDist
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.last_hidden_layer_idx = config.n_layers - hidden_layers
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, position_ids=None, head_mask=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            #token_type_ids=token_type_ids,
                            #position_ids=position_ids,
                            #head_mask=head_mask
                            )

        #print('Outputs shape=', outputs.shape)
        if self.last_hidden_layer_idx == 0:
            sequence_output = outputs[self.last_hidden_layer_idx]
        else:
            sequence_output = outputs[1][self.last_hidden_layer_idx]
        #print("sequence_output type",sequence_output.type())
        pooled_output = sequence_output[:,0,:] #Rekha hack check

        # classification
        pooled_output = self.dropout(pooled_output)
        classifier_logits = self.classifier(pooled_output)

        start_dummy = torch.randn(self.batch_size, self.max_seq_len)
        end_dummy = torch.randn(self.batch_size, self.max_seq_len)
        return start_dummy, end_dummy, classifier_logits