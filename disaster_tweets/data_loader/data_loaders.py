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


class TweetDataset(DataLoader):
    """Loads Tweets from a DataFrame.
    
    Notes
    -----
    The objective of class is preprocessing a series of tweets contained on
    a CSV file so they can be introduced to a model. 
    
    """
    def __init__(self, csv_path, data_preprocessor, batch_size, device="cpu"):
        df = pd.read_csv(csv_path)
        device = torch.device(device)
        data = data_preprocessor(df.text.tolist())
        self.data = self.batchify(data, batch_size, device)

    @staticmethod
    def batchify(data, batch_size, device):
        nbatch = data.size(0) // batch_size
        data = data.narrow(0, 0, nbatch * batch_size)
        data = data.view(batch_size, -1).t().contiguous()
        return data.to(device)

