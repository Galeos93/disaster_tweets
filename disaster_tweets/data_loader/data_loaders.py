from numpy import dtype
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms

from base import BaseDataLoader


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

class DataPreprocessor:
    def __init__(self, tokenizer, vocab, postprocessor):
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.postprocessor = postprocessor

    def __call__(self, sentences_iter):
        tokenized_sentences = [self.tokenizer(x) for x in sentences_iter]
        vectorized_sentences = [torch.tensor(self.vocab(x), dtype=torch.long) 
                                for x in tokenized_sentences]
        return torch.cat(vectorized_sentences)



class TweetDataLoader(DataLoader):
    """Loads Tweets from a DataFrame.
    
    Notes
    -----
    The objective of class is:

    (1) Convert the sentences into a series of tokens
    (2) Convert these tokens into integers, which are part of a Vocabulary
    (3) Postprocess the sentences so they have the same length?
    (4) Batchify them    
    
    """
    def __init__(self, csv_path, data_preprocessor):
        df = pd.read_csv(csv_path)
        data = data_preprocessor(df.text.tolist())
