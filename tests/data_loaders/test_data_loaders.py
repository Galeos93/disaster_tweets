import functools
import pathlib

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from disaster_tweets.data_loader import data_loaders

1


class TestBasicDataPreprocessor:
    @staticmethod
    @pytest.fixture(scope="class")
    def sample_sentences():
        sentences = [
            "Hello how are you?",
            "My name is John Doe\n",
            "Hooow are you doing today?!?",
        ]
        return sentences

    @staticmethod
    def test_given_sentences_output_tensor_has_correct_length(
        sample_sentences,
    ):
        data_preprocessor = data_loaders.BasicDataPreprocessor()
        vectorized_sentences = list(data_preprocessor(sample_sentences))
        assert len(sum(vectorized_sentences, [])) == 14


class TestTweetDataset:
    """Test TweetDataset class.

    Notes
    -----

    The Tweets undergo the
    following transformations:

    1. [x] Tweet cleaner: Emoji removal, url removal, hash removal, etc.
    2. [x] Uppercase to lowercase
    3. [x] Remove punctuation
    4. [ ] Spelling correction
    5. [ ] Stopword removal
    6. [x] Tokenizer

    """

    @staticmethod
    def test_given_real_tweets_output_is_correct():
        tweets_csv = (pathlib.Path("disaster_tweets") / "data" / "train.csv").resolve()
        tweets_df = pd.read_csv(tweets_csv)
        data_preprocessor = data_loaders.BasicDataPreprocessor(
            preprocessor=data_loaders.basic_tweet_preprocessor,
        )
        vocab = data_loaders.VocabBuilder.from_iterator(
            str(tweets_csv),
            data_preprocessor,
        )
        dataset = data_loaders.TweetDataset(tweets_csv, data_preprocessor, vocab=vocab)

        assert len(dataset[0]) == 2
        assert len(dataset[0][1]) > 0
        assert isinstance(dataset[0][0], torch.Tensor)
        assert isinstance(dataset[0][1], torch.Tensor)
        assert len(dataset) == len(tweets_df)


class TestTrainingDataLoader:
    @staticmethod
    def test_given_dataset_batches_are_correct():
        batch_size = 2

        tweets_csv = (pathlib.Path("disaster_tweets") / "data" / "train.csv").resolve()
        data_preprocessor = data_loaders.BasicDataPreprocessor(
            preprocessor=data_loaders.basic_tweet_preprocessor,
        )
        vocab = data_loaders.VocabBuilder.from_iterator(
            str(tweets_csv),
            data_preprocessor,
        )

        dataset = data_loaders.TweetDataset(tweets_csv, data_preprocessor, vocab=vocab)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=data_loaders.collate_batch,
        )
        labels, texts, offsets = next(iter(dataloader))
        assert labels.size()[0] == batch_size
        assert offsets.size()[0] == batch_size
        assert offsets[-1] < texts.size()[0]
        assert labels.dtype == torch.float32
        assert offsets.dtype == torch.int64
        assert texts.dtype == torch.int64
        assert dataset.vocab(["our", "deeds"]) == texts[:2].tolist()
