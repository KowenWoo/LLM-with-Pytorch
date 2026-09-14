import torch
import tiktoken
import regex as re

class Tokenizer():
    def __init__(self, type):
        self.tokenizer = tiktoken.get_encoding(type)

    def encode(self, input):
        return torch.tensor(self.tokenizer.encode(input))

class SimpleTokenizerV1: 
    '''
    Simple tokenizer that splits text into tokens based on whitespace and punctuation.
    It uses a provided vocabulary to map tokens to integer IDs and vice versa.
    '''
    def __init__(self, vocab): 
        self.str_to_int = vocab 
        self.int_to_str = {i:s for s,i in vocab.items()} 
    def encode(self, text): 
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text) 
        preprocessed = [ item.strip() for item in preprocessed if item.strip() ] 
        ids = [self.str_to_int[s] for s in preprocessed] 
        return ids 
    def decode(self, ids): 
        text = " ".join([self.int_to_str[i] for i in ids]) 
        # Replace spaces before the specified punctuations 
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) 
        return text