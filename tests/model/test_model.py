import functools
from os import stat
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
            outputs = model_instance(sentence, offset)
            len(outputs) == 32


class TestCNNModel:
    @staticmethod
    def get_activation(name, activation):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    def test_given_input_output_shape_is_correct(self):
        batch_size = 8
        length = 16
        embedding_size = 32

        vocab = list(range(128))

        input = torch.randint(0, 128, size=(length, batch_size))

        model_instance = model.CNNModel(embedding_size, vocab)

        activations = {}

        model_instance.conv_1d_16.register_forward_hook(
            self.get_activation("conv_1d_16", activation=activations)
        )
        model_instance.conv_1d_8.register_forward_hook(
            self.get_activation("conv_1d_8", activation=activations)
        )
        model_instance.conv_1d_4.register_forward_hook(
            self.get_activation("conv_1d_4", activation=activations)
        )

        model_instance.max_pool_1.register_forward_hook(
            self.get_activation("max_pool_1", activation=activations)
        )
        model_instance.max_pool_2.register_forward_hook(
            self.get_activation("max_pool_2", activation=activations)
        )
        model_instance.max_pool_3.register_forward_hook(
            self.get_activation("max_pool_3", activation=activations)
        )
        output = model_instance(input)

        assert activations["conv_1d_16"].shape == (batch_size, 18, 1)
        assert activations["conv_1d_8"].shape == (batch_size, 9, 9)
        assert activations["conv_1d_4"].shape == (batch_size, 5, 13)

        assert activations["max_pool_1"].shape == (batch_size, 5, 1)
        assert activations["max_pool_2"].shape == (batch_size, 9, 1)
        assert activations["max_pool_3"].shape == (batch_size, 18, 1)

        assert output.shape == (8, 1)


class TestLSTMModel:
    @staticmethod
    def test_given_data_output_shape_is_correct():
        batch_size = 8
        length = 16
        embedding_size = 32

        vocab = list(range(128))
        model_instance = model.LSTMModel(
            embedding_size=embedding_size,
            vocab=vocab,
        )
        input = torch.randint(0, 128, size=(length, batch_size))
        with torch.no_grad():
            outputs = model_instance(input)
            assert outputs.shape == (8, 1)
