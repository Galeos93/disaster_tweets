from os import stat
import torch

from disaster_tweets.model import metric


class TestAccuracy:
    @staticmethod
    def test_given_input_output_is_correct():
        target = torch.tensor([1, 1, 1, 1])
        output = torch.tensor([-10, -10, 10, 10])
        output_metric = metric.accuracy(output, target)
        assert output_metric == 0.5
