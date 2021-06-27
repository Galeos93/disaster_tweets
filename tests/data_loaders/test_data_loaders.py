import pathlib

import pandas as pd
import pytest
import torch
from torchtext.data.utils import get_tokenizer

from disaster_tweets.data_loader import data_loaders
from disaster_tweets.utils import nlp


@pytest.fixture()
def preprocessor():
    def inner_fun(sentence):
        sentence = nlp.tweet_cleaner(sentence)
        sentence = sentence.lower()
        sentence = nlp.remove_punct(sentence)
        return sentence

    return inner_fun


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
        "tokenizer,expected_length", [(get_tokenizer("basic_english"), 14)]
    )
    def test_given_sentences_output_tensor_has_correct_length(
        tokenizer,
        expected_length,
        sample_sentences,
        preprocessor,
    ):
        data_preprocessor = data_loaders.DataPreprocessor(
            tokenizer, preprocessor, lambda x: x
        )
        vectorized_sentences = list(data_preprocessor(sample_sentences))
        assert len(sum(vectorized_sentences, [])) == expected_length


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
    def test_given_real_tweets_output_is_correct(preprocessor):
        tweets_csv = (pathlib.Path("disaster_tweets") / "data" / "train.csv").resolve()
        tweets_df = pd.read_csv(tweets_csv)
        tokenizer = get_tokenizer("basic_english")
        data_preprocessor = data_loaders.DataPreprocessor(
            tokenizer=tokenizer,
            preprocessor=preprocessor,
            postprocessor=lambda x: x,
        )
        dataset = data_loaders.TweetDataset(tweets_csv, data_preprocessor)
        
        assert len(dataset[0]) == 3
        assert len(dataset[0][1]) > 0
        assert isinstance(dataset[0][0], torch.Tensor)
        assert isinstance(dataset[0][1], torch.Tensor)
        assert isinstance(dataset[0][2], torch.Tensor)
        assert len(dataset[0][1]) == dataset[0][-1]
        assert len(dataset) == len(tweets_df)
