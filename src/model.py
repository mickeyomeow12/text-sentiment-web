import config
import transformers
import torch.nn as nn


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        # print(self.bert)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        # _, o2, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        _, o2 = self.bert(
            ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False
        )

        # _, o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        bo = self.bert_drop(o2)
        output = self.out(bo)
        return output
