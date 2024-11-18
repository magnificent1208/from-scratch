import math
import time
import spacy
import torch

from torch import nn, optim
from torch.optim import Adam
from torch import tensor

from torchtext.legacy.data import Field, BucketIterator
from torchtext.datasets import Multi30k

#tokenizer 英文德文的tokeniizer 训练翻译模型

class Tokenizer:
    def __init__(self):
        self.spacy_de = spacy.load('de_core_news_sm') 
        self.spacy_en = spacy.load('en_core_web_sm') 

    def tokenizer_de(self,text):
        return [tok.text for tok in self.spacy_de.tokeizer(text)]
    
    def tokenizer_en(self,text):
        return [tok.text for tok in self.spacy_en.tokenizer(text)]
    
tokenizer = Tokenizer()
example = 'Hello from the other side.'
tokens = tokenizer.tokenizer_en(example)

print(example)
print(tokens)

#DataLoader
class DataLoader:
    source: Field = None
    target: Field = None
    def __init__(self,ext,tokenizer_en, tokenizer_de, init_token, eos_token):
        self.ext = ext
        self.tokenizer_en = tokenizer_en
        self.tokenizer_de = tokenizer_de
        self.init_token = init_token
        self.eos_token = eos_token
    
    def make_dataset(self):
        if self.ext == ('.de', '.en'):
            self.source = Field(tokenizer=self.tokenizer_de, init_token=self.init_token,eos_token=self.eos_token,
                                lower=True, batch_first=True)
            self.target =Field(tokenizer=self.tokenizer_en, init_token=self.init_token,eos_token=self.eos_token,
                                lower=True, batch_first=True)
        elif self.ext == ('.en', '.de'):
            self.source = Field(tokenizer=self.tokenizer_en, init_token=self.init_token,eos_token=self.eos_token,
                                lower=True, batch_first=True)
            self.target =Field(tokenizer=self.tokenizer_de, init_token=self.init_token,eos_token=self.eos_token,
                                lower=True, batch_first=True)
            
        #拆分数据集
        train_data, valid_data,test_data, = Multi30k()
        return train_data, valid_data, test_data        
    
    def build_vocab(self, train_data, min_freq):
        self.source.build_vocab(train_data,min_freq-min_freq)
        self.target.build_vocab(train_data,min_freq=min_freq)

    def make_iter(self, train, validate, test, batch_size, device):
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train, validate, test),
                                                                              batch_size=batch_size,
                                                                              device=device)
        print('dataset initializing done')
        return train_iterator, valid_iterator, test_iterator

                       
    
        