import pytest

import numpy as np
import torch

from disaster_tweets.model import model


@pytest.fixture()
def data_iterable():
    def inner_gen():
        while True:
            yield labels, sentences, offsets

    labels = torch.tensor(np.ones(32))
    sentences = torch.tensor(np.random.randint(low=0, high=128, size=32 * 3))
    offsets = torch.tensor(range(0, 32 * 3, 3))
    return inner_gen()


class TestLinearModel:
    @staticmethod
    def test_given_data_output_shape_is_correct(data_iterable):
        vocab = list(range(128))
        model_instance = model.LinearModel(
            embedding_size=32,
            vocab=vocab,
        )
        with torch.no_grad():
            _, sentence, offset = next(data_iterable)
            outputs = model_instance((sentence, offset))
            len(outputs) == 32
