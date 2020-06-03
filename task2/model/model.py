import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

from transformers import BertForQuestionAnswering

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class FinCausalBert(BaseModel):
    def __init__(self):
        super().__init__()
        self.cause_bert = BertForQuestionAnswering.from_pretrained('bert-base-cased')
        self.effect_bert = BertForQuestionAnswering.from_pretrained('bert-base-cased')
        # self.fc = nn.Sequential(
        #                         nn.Linear(768, 256),
        #                         nn.ReLU(),
        #                         nn.Linear(256, 128),
        #                         nn.ReLU(),
        #                         nn.Linear(128, 4),
        #                          )
       
    def forward(self, input_ids, cause_start=None, cause_end=None, effect_start=None, effect_end=None):
        cause_output = self.cause_bert(input_ids=input_ids, start_positions=cause_start, end_positions=cause_end)
        effect_output = self.effect_bert(input_ids=input_ids, start_positions=effect_start, end_positions=effect_end)
        if(cause_start != None): 
            loss = cause_output[0] + effect_output[0]
            score = {'cause_start': cause_output[1], 'cause_end': cause_output[2], 'effect_start': effect_output[1], 'effect_end':effect_output[2]}
            return loss, score
        else: 
            score = {'cause_start': cause_output[0], 'cause_end': cause_output[1], 'effect_start': effect_output[0], 'effect_end':effect_output[1]}
            return score
