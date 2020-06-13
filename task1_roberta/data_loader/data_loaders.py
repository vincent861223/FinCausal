from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
from transformers import BertTokenizer
from transformers import RobertaTokenizer


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
    def __init__(self, datapath, train=False, test=False):
        self.datapath = datapath
        self.data = []
        self.train = train
        self.test = test
        #self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)
        self.loadData(datapath, train=self.train, test=self.test)
        

    def __len__(self):
        return len(self.data)

    def  __getitem__(self, index):
        return self.data[index]

    def loadData(self, datapath, train=False, test=False):
        with open(datapath) as f:
            lines = f.readlines()[1:]
            lines = tqdm(lines, desc="Loading data")
            for line in lines:
                token = line[:-1].split(";")
                if train:
                    self.data.append({"id": token[0], "text": token[1], "index": self.sentenceToIndex(token[1]), "gold": int(token[2])})
                if test:
                    self.data.append({"id": token[0], "text": token[1], "index": self.sentenceToIndex(token[1])})

    def sentenceToIndex(self, sentence, add_special_tokens=True):
        try:
            return self.tokenizer.encode(text=sentence, add_special_tokens=add_special_tokens, max_length=256, pad_to_max_length=True)
        except :
            return self.prepare_features(sentence)
        #return self.tokenizer.encode(text=sentence, add_special_tokens=add_special_tokens, max_length=256, pad_to_max_length=True)
        #return self.prepare_features(sentence)

    def collate_fn(self, datas):
        batch = {} 
        batch["id"] = [data["id"] for data in datas]
        batch["text"] = [data["text"] for data in datas]
        batch["index"] = torch.tensor([data["index"] for data in datas])
        if self.train: batch["gold"] = torch.tensor([data["gold"] for data in datas])
        return batch

    def prepare_features(self,seq_1, max_seq_length = 256, 
                 zero_pad = True, include_CLS_token = True, include_SEP_token = True):
        ## Tokenzine Input
        tokens_a = self.tokenizer.tokenize(seq_1)

        ## Truncate
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]
        ## Initialize Tokens
        tokens = []
        if include_CLS_token:
            tokens.append(self.tokenizer.cls_token)
        ## Add Tokens and separators
        for token in tokens_a:
            tokens.append(token)

        if include_SEP_token:
            tokens.append(self.tokenizer.sep_token)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ## Zero-pad sequence lenght
        if zero_pad:
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
        return input_ids
            

