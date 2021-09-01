import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class LinearModel(nn.Module):
    def __init__(self, embedding_size, vocab):
        super(LinearModel, self).__init__()
        self.embedding_layer = nn.EmbeddingBag(len(vocab), embedding_size)
        self.dense_0 = nn.Linear(embedding_size, 10)
        self.dense_1 = nn.Linear(10, 1)

    def forward(self, inputs, offsets):
        embeddings = self.embedding_layer(inputs, offsets)  # B, E
        outputs = self.dense_0(embeddings)
        outputs = self.dense_1(outputs)
        return outputs


class CNNModel(nn.Module):
    def __init__(self, embedding_size, vocab):
        super(CNNModel, self).__init__()
        self.embedding_layer = nn.Embedding(len(vocab), embedding_size)
        self.conv_1d_16 = nn.Conv1d(embedding_size, 18, kernel_size=16)
        self.conv_1d_8 = nn.Conv1d(embedding_size, 9, kernel_size=8)
        self.conv_1d_4 = nn.Conv1d(embedding_size, 5, kernel_size=4)
        self.max_pool_1 = nn.AdaptiveMaxPool1d(1)
        self.max_pool_2 = nn.AdaptiveMaxPool1d(1)
        self.max_pool_3 = nn.AdaptiveMaxPool1d(1)
        self.dropout_1 = nn.Dropout(p=0.4)
        self.dropout_2 = nn.Dropout(p=0.4)
        self.dropout_3 = nn.Dropout(p=0.4)
        # Classifier
        self.dense_1 = nn.Linear(embedding_size, 1)

    def forward(self, x):
        # Input: L, B, 1
        # Transformation: L*B, 1
        length, batch_size = x.shape
        x = x.view(-1, 1)
        x = self.embedding_layer(x)
        # Back to original dimension: L, B, E
        x = x.view(length, batch_size, -1)
        # Reshape to B, E, L
        x = x.transpose(0, 1)
        x = x.transpose(1, 2)
        x_1 = self.conv_1d_4(x)  # Requires B, E, L. Returns B, E_1, L_1
        x_2 = self.conv_1d_8(x)  # Requires B, E, L. Returns B, E_2, L_2
        x_3 = self.conv_1d_16(x)  # Requires B, E, L. Returns B, E_3, L_3

        x_1 = self.dropout_1(x_1)
        x_2 = self.dropout_1(x_2)
        x_3 = self.dropout_1(x_3)

        x_1 = self.max_pool_1(x_1)  # Returns B, E_1
        x_2 = self.max_pool_2(x_2)  # Returns B, E_2
        x_3 = self.max_pool_3(x_3)  # Returns B, E_3

        x = torch.cat((x_1, x_2, x_3), dim=1)  # Returns B, E, L_1 + L_2 + L_3
        x = torch.squeeze(x)
        x = self.dense_1(x)
        return x


class LSTMModel(nn.Module):
    def __init__(self, embedding_size, vocab):
        super(LSTMModel, self).__init__()
        self.embedding_layer = nn.Embedding(len(vocab), embedding_size)
        self.lstm = nn.LSTM(embedding_size, embedding_size * 2, 1)
        self.dense_0 = nn.Linear(embedding_size * 2, 10)
        self.avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.dense_1 = nn.Linear(64, 1)

    def forward(self, x):
        length, batch_size = x.shape
        x = x.view(-1, 1)
        x = self.embedding_layer(x)
        x = x.view(length, batch_size, -1)
        output, (hn, cn) = self.lstm(x)
        x = torch.mean(output, 0)
        x = self.dense_1(x)
        return x
