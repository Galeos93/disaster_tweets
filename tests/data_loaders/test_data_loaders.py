import pathlib

import pandas as pd
import pytest
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from disaster_tweets.data_loader import data_loaders
from disaster_tweets import data


class TestDataPreprocessor:
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
    @pytest.mark.parametrize(
        "tokenizer,expected_length", [(get_tokenizer("basic_english"), 18)]
    )
    def test_given_sentences_output_tensor_has_correct_length(
        tokenizer,
        expected_length,
        sample_sentences,
    ):
        vocab = build_vocab_from_iterator(
            map(tokenizer, sample_sentences), specials=["<unk>"]
        )
        data_preprocessor = data_loaders.DataPreprocessor(tokenizer, vocab, lambda x: x)
        vectorized_sentences = data_preprocessor(sample_sentences)

        assert len(vectorized_sentences) == expected_length


class TestTweetDataset:
    @staticmethod
    @pytest.fixture(scope="class")
    def data_preprocessor(vocab):
        return data_loaders.DataPreprocessor(
            tokenizer=get_tokenizer("basic_english"),
        )

    @staticmethod
    def test_given_real_tweets_output_is_correct():
        tweets_csv = (pathlib.Path("disaster_tweets") / "data" / "train.csv").resolve()
        tweets_df = pd.read_csv(tweets_csv)
        tokenizer = get_tokenizer("basic_english")
        vocab = build_vocab_from_iterator(
            map(tokenizer, tweets_df.text.tolist()), specials=["<unk>"]
        )
        data_preprocessor = data_loaders.DataPreprocessor(
            tokenizer=tokenizer, vocab=vocab, postprocessor=lambda x: x
        )
        dataset = data_loaders.TweetDataset(tweets_csv, data_preprocessor, 32)
        assert dataset.data.shape[-1] == 32
