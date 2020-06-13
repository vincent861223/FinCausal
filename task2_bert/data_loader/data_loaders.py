from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
from transformers import BertTokenizer
import os

os.environ['TORCH_HOME'] = '/tmp3/b05505019/cache'

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class FinCausalDataloader(BaseDataLoader):
    def __init__(self, datapath, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.datapath = datapath
        self.training = training
        self.dataset = FinCausalDataset(datapath, train=training, test=(not training))
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=self.dataset.collate_fn)
 

class FinCausalDataset(Dataset):
    def __init__(self, datapath, train=False, test=False, mode='Cause'):
        self.datapath = datapath
        self.data = []
        self.train = train
        self.test = test
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.special_tokens = ['<e1>', '</e1>', '<e2>', '</e2>'] 
        self.tokenizer.add_tokens(self.special_tokens)
        self.max_label = self.maxLabel()
        self.mode = mode
        self.loadData(datapath, train=self.train, test=self.test)
        print(self.max_label)
        

    def __len__(self):
        return len(self.data)

    def  __getitem__(self, index):
        return self.data[index]

    def loadData(self, datapath, train=False, test=False):
        with open(datapath) as f:
            lines = f.readlines()
            columns = lines[0][:-1].split(';')
            lines = lines[1:]
            lines = tqdm(lines, desc="Loading data")
            for line in lines:
                token = line[:-1].split(";")
                self.data.append({column: token for column, token in zip(columns, token)})

            for i in tqdm(range(len(self.data)), desc='indexing sentences'):
                self.data[i]['indexed_text'] = self.sentenceToIndex(self.data[i]['Text'])
                self.data[i]['tokened_text'] = self.tokenizer.convert_ids_to_tokens(self.data[i]['indexed_text'])

            if train:
                for i in tqdm(range(len(self.data)), desc='converting charPos to wordPos'):
                    self.data[i]['Cause_Start'], self.data[i]['Cause_End'], self.data[i]['Effect_Start'], self.data[i]['Effect_End'] = self.charPos2WordPos(self.data[i]['Sentence'], self.data[i]['indexed_text'])

    def charPos2WordPos(self, text, indexed_text):
        tokens = self.sentenceToIndex(text) 
        cause_start = tokens.index(self.tokenizer.get_vocab()['<e1>'])
        cause_end = tokens.index(self.tokenizer.get_vocab()['</e1>'])
        effect_start = tokens.index(self.tokenizer.get_vocab()['<e2>'])
        effect_end = tokens.index(self.tokenizer.get_vocab()['</e2>'])
        
        if cause_start < effect_start:
            cause_end -= 2
            effect_start -= 2
            effect_end -= 4
        else :
            cause_start -= 2
            cause_end -= 4
            effect_end -=2

        return cause_start, cause_end, effect_start, effect_end

    def maxLabel(self):
        max_label = 0
        checks = ['Cause_Start', 'Cause_End', 'Effect_Start', 'Effect_End']
        for check in checks:
            for data in self.data:
                offset = int(data[check])
                if offset > max_label: max_label = offset
        return max_label


    def sentenceToIndex(self, sentence, add_special_tokens=True):
        return self.tokenizer.encode(text=sentence, add_special_tokens=add_special_tokens, pad_to_max_length=True)

    def collate_fn(self, datas):
        batch = {} 
        batch["id"] = [data["Index"] for data in datas]
        batch["text"] = [data["Text"] for data in datas]
        batch['tokened_text'] = [data['tokened_text'] for data in datas]
        batch["input_ids"] = torch.tensor([data["indexed_text"] for data in datas])
        if self.train:
            batch["cause_start"] = torch.tensor([data["Cause_Start"] for data in datas])
            batch["cause_end"] = torch.tensor([data["Cause_End"] for data in datas])
            batch["effect_start"] = torch.tensor([data["Effect_Start"] for data in datas])
            batch["effect_end"] = torch.tensor([data["Effect_End"] for data in datas])
        return batch
            

