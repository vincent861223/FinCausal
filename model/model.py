import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

from transformers import BertModel

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
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Sequential(
                                nn.Linear(768, 256),
                                nn.ReLU(),
                                nn.Linear(256, 128),
                                nn.ReLU(),
                                nn.Linear(128, 1),
                                nn.Sigmoid()
                                 )
       
    def forward(self, x):
        output = self.bert(x)
        output = output[0].mean(1)
        output = self.fc(output)
        return output
