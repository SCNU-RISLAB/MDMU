import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mamba_ssm import Mamba2, Mamba

class Configs:
    enc_in = 1024 + 1024 + 768
    seq_len = 96
    pred_len = 96
    individual = False  # use independent linear layers for each channel or not

class Mambalayer_seq(nn.Module):
    def __init__(self,len):
        super(Mambalayer_seq, self).__init__()
        self.mamba = Mamba(
            d_model=len,
            d_state=16,
            d_conv = 4,
            expand=2,
        ).to("cuda")

    def forward(self, x):
        x = self.mamba(x)
        return x

class Mambalayer_fea(nn.Module):
    def __init__(self,dim):
        super(Mambalayer_fea, self).__init__()
        self.mamba = Mamba(
            d_model=dim,
            d_state=16,
            d_conv = 4,
            expand=2,
        ).to("cuda")

    def forward(self, x):
        x = self.mamba(x)
        return x

class Mambalayer_fea_down(nn.Module):
    def __init__(self,dim):
        super(Mambalayer_fea_down, self).__init__()
        self.mamba = Mamba(
            d_model=dim,
            d_state=16,
            d_conv = 4,
            expand=2,
        ).to("cuda")

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.mamba(x)
        x = x.permute(0,2,1)
        return x

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class block_model(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, input_channels, input_len, out_len, individual, dimension):
        super(block_model, self).__init__()
        self.channels = input_channels
        self.input_len = input_len
        self.out_len = out_len
        self.individual = individual
        self.dimension=dimension
        self.dropout = nn.Dropout(p=0.5)

        if self.individual:
            self.Linear_channel = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_channel.append(nn.Linear(self.input_len, self.out_len))
        else:
            self.Linear_channel = nn.Linear(self.input_len, self.out_len)
        self.ln = nn.LayerNorm(out_len)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [Batch, Channel, Input length]
        if self.individual:
            output = torch.zeros([x.size(0), x.size(1), self.out_len], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, i, :] = self.Linear_channel[i](x[:, i, :])
        else:
            if self.dimension=="feature":
                output = self.Linear_channel(x.permute(0, 2, 1))
                output = output.permute(0, 2, 1)
            else:  # dimension=="sequence"
                output = self.Linear_channel(x)
        # output = self.ln(output)
        # output = self.relu(output)
        return output  # [Batch, Channel, Output length]

class Gate_unit(nn.Module):
    def __init__(self, configs, t1_dim, t2_dim ):
        super(Gate_unit, self).__init__()

        self.input_channels = configs.enc_in
        # linear layers to generate fusion weights
        self.weight_generator = nn.Linear(t2_dim*2, t2_dim)
        self.sigmoid = nn.Sigmoid()
        self.change_t1dim = nn.Linear(t1_dim, t2_dim)
        self.DoubleBack_dim = nn.Linear(t2_dim*2, t2_dim)
        self.change_fuseddim = nn.Linear(t2_dim, t2_dim//2)
        # self.mamba = Mambalayer_fea(t2_dim//2)
        self.mamba = Mambalayer_fea(t2_dim)

        # Upsampling convolutional layer
        self.upconv = nn.ConvTranspose1d(
            in_channels=t2_dim, out_channels=t2_dim, kernel_size=2, stride=2
        )


    def forward(self, t1, t2, t1_len):
        if t1.size(1) != t2.size(1):
            t1 = t1.permute(0, 2, 1)
            t1 = self.change_t1dim(t1)
            t1 = t1.permute(0, 2, 1)

        # t2= t2.unsqueeze(2)
        # t2 = F.interpolate(t2, size=(1, t1_len), mode='nearest')
        # t2 = t2.squeeze(2)  # (16,4096,24)
        t2 = self.upconv(t2)

        fused = torch.cat((t1, t2), dim=1)  # Concatenate the e3 feature map with the d4 feature map in the feature channel,
        # (16, 8192, 48)

        weights = self.sigmoid(self.weight_generator(fused.permute(0, 2, 1)))
        weights = weights.permute(0, 2, 1)
        weighted_t1 = t1 * weights
        weighted_t2 = t2 * (1 - weights)

        # weighted_t1 = self.dropout(weighted_t1)
        # weighted_t2 = self.dropout(weighted_t2)

        fused = torch.cat([weighted_t1, weighted_t2], dim=1)
        fused = fused.permute(0, 2, 1)
        fused = self.DoubleBack_dim(fused)
        # fused = self.change_fuseddim(fused)
        fused = self.mamba(fused)
        fused = fused.permute(0, 2, 1)


        return fused

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.input_channels = configs.enc_in
        self.input_len = configs.seq_len
        self.out_len = configs.pred_len
        self.individual = configs.individual
        self.dropout = nn.Dropout(p=0.5)

        # Downsampling setting
        n1 = 1
        filters = [n1, n1 * 2, n1 * 4, n1 * 8]
        self.down_in = [int(self.input_len / filters[0]), int(self.input_len / filters[1]), int(self.input_len / filters[2]),
                   int(self.input_len / filters[3])]
        self.down_out = [int(self.out_len / filters[0]), int(self.out_len / filters[1]), int(self.out_len / filters[2]),
                    int(self.out_len / filters[3])]


        self.Maxpool1 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.Maxpool2 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.Maxpool3 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.Maxpool4 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)

        self.conv1 = nn.Conv1d(in_channels=self.input_channels, out_channels=self.input_channels, kernel_size=3,
                               stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=self.input_channels, out_channels=self.input_channels, kernel_size=3,
                               stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=self.input_channels, out_channels=self.input_channels, kernel_size=3,
                               stride=2, padding=1)

        self.down_block1 = Mambalayer_fea_down(self.input_channels)
        self.down_block2 = Mambalayer_fea_down(self.input_channels)
        self.down_block3 = Mambalayer_fea_down(self.input_channels)
        self.down_block4 = Mambalayer_fea_down(self.input_channels)

        self.gate4 = Gate_unit(configs, self.input_channels, self.input_channels)
        self.gate3 = Gate_unit(configs, self.input_channels, self.input_channels)
        self.gate2 = Gate_unit(configs, self.input_channels, self.input_channels)


    def forward(self, x):
        x1 = x.permute(0,2,1)  # [bs, dim, seq_len]
        e1 = self.down_block1(x1)

        x2 = self.conv1(x1)  # [bs,dim,seq_len=48]
        e2 = self.down_block2(x2)

        x3 = self.conv2(x2)  # [bs,dim,seq_len=24]
        e3 = self.down_block3(x3)

        x4 = self.conv3(x3)  # [bs,dim,seq_len=12]
        e4 = self.down_block4(x4)


        d4 = self.gate4(e3, e4, self.down_out[2])  # [batch_size, input_channels*2, 24]
        # d4 = self.up_block3(d4)  # [batch_size, input_channels, 24]

        d3 = self.gate3(e2, d4, self.down_out[1])  # [batch_size, input_channels*2, 48]
        # d3 = self.up_block2(d3)  # [batch_size, input_channels, 48]

        d2 = self.gate2(e1, d3, self.down_out[0])  # [batch_size, input_channels*2, 96]
        # out = self.up_block1(d2)  # [batch_size, input_channels, 96]
        out = d2

        # out = self.linear_out(d2)

        return out.permute(0, 2, 1)