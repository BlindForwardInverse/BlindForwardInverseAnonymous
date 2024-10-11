import torch
import torch.nn as nn

class TFRNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, hidden_ch=128):
        super().__init__()
        self.freq = FrequencyRNNBlock(in_ch, hidden_ch, out_ch)
        self.time = TimeRNNBlock(in_ch, hidden_ch, out_ch)

    def forward(self, x):
        x, skip = self.freq(x)
        x = self.time(x, skip)
        return x

class FrequencyRNNBlock(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, bidirectional=True, rnn_type='gru', num_layers=1):
        super(FrequencyRNNBlock, self).__init__()

        self.h = hidden_ch
        self.o = out_ch

        if bidirectional:
            if rnn_type == 'gru':
                self.rnn = torch.nn.GRU(input_size=in_ch, hidden_size=int(hidden_ch // 2), num_layers=num_layers,
                                        batch_first=True, bidirectional=True, bias=True)
            elif rnn_type == 'lstm':
                self.rnn = torch.nn.LSTM(input_size=in_ch, hidden_size=int(hidden_ch // 2), num_layers=num_layers,
                                         batch_first=True, bidirectional=True)
            else:
                self.rnn = torch.nn.RNN(input_size=in_ch, hidden_size=int(hidden_ch // 2), num_layers=num_layers,
                                        batch_first=True, bidirectional=True)
        else:
            if rnn_type == 'gru':
                self.rnn = torch.nn.GRU(input_size=in_ch, hidden_size=hidden_ch, num_layers=num_layers,
                                        batch_first=True, bidirectional=False, bias=True)
            elif rnn_type == 'lstm':
                self.rnn = torch.nn.LSTM(input_size=in_ch, hidden_size=hidden_ch, num_layers=num_layers,
                                         batch_first=True, bidirectional=False)
            else:
                self.rnn = torch.nn.RNN(input_size=in_ch, hidden_size=hidden_ch, num_layers=num_layers,
                                        batch_first=True, bidirectional=False)

        self.fc = nn.Conv2d(hidden_ch, out_ch, 1, 1, bias=True)
        # self.bn = nn.SyncBatchNorm(out_ch)
        self.bn = nn.BatchNorm2d(out_ch)
        # self.relu = nn.ReLU6()
        self.relu = nn.GELU()

    def forward(self, input):
        x = input
        # 1. Frequency-wise LSTM

        # Batch, Channel, Frequency, Time -> Batch*Time, Frequency, Channel
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3)  # Batch, Channel, Time, Frequency -> Batch, Time, Channel, Frequency
        x = x.reshape(b * t, c, f)  # Batch, Time, Channel, Frequency -> Batch*Time, Channel, Frequency
        x = x.permute(0, 2, 1)  # Batch*Time, Channel, Frequency -> Batch*Time, Frequency, Channel
        x, _ = self.rnn(x)
        x = x.reshape(b, t, f, self.h)  # Batch*Time, Frequency, Channel -> Batch, Time, Frequency, Channel
        x = x.permute(0, 3, 1, 2)  # Batch, Time, Frequency, Channel -> Batch, Channel, Time, Frequency

        # 2. Linear
        skip = self.fc(x)

        # 3. Activation
        x = self.bn(skip)
        x = self.relu(x)

        return x, skip


class TimeRNNBlock(nn.Module):
    def __init__(self, in_ch=256, hidden_ch=256, out_ch=256, num_layers=1, bidirectional=False):
        super(TimeRNNBlock, self).__init__()
        self.h = hidden_ch
        self.o = out_ch

        if bidirectional:
            self.rnn = torch.nn.GRU(input_size=in_ch, hidden_size=int(hidden_ch // 2), num_layers=num_layers,
                                    batch_first=True, bidirectional=True, bias=True)
        else:
            self.rnn = torch.nn.GRU(input_size=in_ch, hidden_size=hidden_ch, num_layers=num_layers, batch_first=True,
                                    bidirectional=False, bias=True)

        # self.fc = nn.Linear(hidden_ch, out_ch, bias=True)
        self.fc = nn.Conv2d(hidden_ch, out_ch, 1, 1, bias=True)
        # self.bn = nn.SyncBatchNorm(out_ch)
        self.bn = nn.BatchNorm2d(out_ch)
        # self.relu = nn.ReLU6()
        self.relu = nn.GELU()

    def forward(self, input, skip):
        x = input

        # Channel-wise LSTM
        b, c, t, f = x.shape  # Batch, Channel, Time, Frequency
        x = x.permute(0, 3, 2, 1)  # Batch, Channel, Frequency, Time -> Batch, Frequency, Time, Channel
        x = x.reshape(b * f, t, c)  # Batch, Frequency, Time, Channel -> Batch*Frequency, Time, Channel
        x, _ = self.rnn(x)
        # x = self.fc(x)
        x = x.reshape(b, f, t, self.h)  # Batch*Frequency, Time, Channel -> Batch, Frequency, Time, Channel
        x = x.permute(0, 3, 2, 1)  # Batch, Frequency, Time, Channel -> Batch, Channel, Time, Frequency
        x = self.fc(x) + skip
        x = self.bn(x)
        x = self.relu(x)

        return x
