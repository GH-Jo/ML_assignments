# -*- coding: utf-8 -*-

import torch
from torch import nn
from bert.bert_model import BertPreTrainedModel, BertModel

class BERT_MLP(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        # ========== [ 여기를 구현하세요. ] ==========
        # Multi Layer Perceptron 구현
        self.layer1=nn.Linear(768*512, 200)
        self.layer2=nn.Linear(200, 512*11)
        
        # ========== [ 여기까지 구현하세요. ] ==========
        # BERT init
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        loss_fct = nn.CrossEntropyLoss()
        loss = None
        output = None
        # ========== [ 여기를 구현하세요. ] ==========
        # self.bert를 사용하여 bert output 생성 및 MLP 구현
        # CrossEntropyLoss를 이용하여 Loss 계산
        batch_size = input_ids.shape[0]
        def print_shape(varname, var):
          print('{}.shape : {}'.format(varname, var.shape))

        x, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        x = x.view(batch_size, -1)

        x = self.layer1(x)
        x = self.layer2(x)
        output = x.view(batch_size*512, 11)
        labels = labels.view(batch_size*512)
        loss = loss_fct(output, labels)
        output = x.view(batch_size, 512, 11)
        


        # ========== [ 여기까지 구현하세요. ] ==========
        return loss, output

        