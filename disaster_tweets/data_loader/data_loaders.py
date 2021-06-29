from dataclasses import dataclass
import typing

import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchtext.vocab import build_vocab_from_iterator
from torchvision import datasets, transforms
from torchtext.data.utils import get_tokenizer

from base import BaseDataLoader


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        training=True,
    ):
        trsfm = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(
            self.data_dir, train=training, download=True, transform=trsfm
        )
        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )


@dataclass
class _BaseDataPreprocessor:
    """This class transforms an iterable of sentences.

    Notes
    -----
    The sentences fed to the class instance will be preprocessed, tokenized
    and postprocessed. These three stages and configurable.

    """

    tokenizer: typing.Callable[[str], typing.List[str]] = None
    preprocessor: typing.Callable[[str], str] = lambda x: x

    def __call__(self, sentences_iter):
        sentences_iter = [self.preprocessor(sentence) for sentence in sentences_iter]
        tokenized_sentences = (self.tokenizer(x) for x in sentences_iter)
        return tokenized_sentences


class BasicDataPreprocessor(_BaseDataPreprocessor):
    tokenizer: get_tokenizer("basic_english")


class TweetDataset(Dataset):
    """Loads Tweets from a DataFrame.

    Notes
    -----
    The objective of class is preprocessing a series of tweets contained on
    a CSV file so they can be introduced to a model.

    """

    def __init__(self, csv_path, data_preprocessor):
        df = pd.read_csv(csv_path)
        tokenized_sentences = list(data_preprocessor(df.text.tolist()))
        self.vocab = build_vocab_from_iterator(tokenized_sentences, specials=["<unk>"])
        sequences = [
            torch.tensor(self.vocab(x), dtype=torch.long) for x in tokenized_sentences
        ]
        labels = torch.tensor(df.target.astype(int))
        offsets = torch.tensor([len(x) for x in sequences])
        self.data = list(zip(labels, sequences, offsets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx]
