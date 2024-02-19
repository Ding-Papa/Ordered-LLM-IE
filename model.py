import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import math
from RL_utils.gtrxl import GTrXL

class ActorModel(nn.Module):
    def __init__(self, plm_name, **kwargs):
        super(ActorModel, self).__init__(**kwargs)
        self.bert = AutoModel.from_pretrained(plm_name)
        self.output_layer = nn.Linear(768, 1)
    
    def forward(self, input_ids, token_type_ids):
        last_hidden_state = self.bert(input_ids=input_ids, token_type_ids=token_type_ids).last_hidden_state
        logits = self.output_layer(last_hidden_state[:,0,:])
        return logits


class ActorModel_grtxl(nn.Module):
    def __init__(self, plm_name, **kwargs):
        super(ActorModel_grtxl, self).__init__(**kwargs)
        self.bert = AutoModel.from_pretrained(plm_name)
        self.gtrxl = GTrXL(
            input_dim = 768,
            embedding_dim = 768
            )
        self.output_layer = nn.Linear(768, 1)

    def forward(self, input_ids, token_type_ids, choice_mask):
        last_hidden_state = self.bert(input_ids=input_ids, token_type_ids=token_type_ids).last_hidden_state
        gtrxl_output = self.gtrxl(last_hidden_state)

        logits = self.output_layer(gtrxl_output['logit']).squeeze(-1)
        logits = logits.masked_fill_((1 - choice_mask).bool(), -1e12)
        return logits
