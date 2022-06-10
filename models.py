import math
import torch
import torch.nn as nn
from typing import List, Optional

def linear_softmax_pooling(x):
    return (x ** 2).sum(1) / x.sum(1)

class Crnn(nn.Module):
    def __init__(self, num_freq, class_num, dropout, enlarge: int =1):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     num_freq: int, mel frequency bins
        #     class_num: int, the number of output classes
        ##############################
        super().__init__()
        self.num_freq = num_freq
        self.class_num = class_num
        self.dropout = dropout
        self.batch_norm = nn.BatchNorm1d(64)
        self.conv1 = Conv_Block(1, 16 , 3 , 4)
        self.conv2 = Conv_Block(16, 32 , 3, 4)
        self.conv3 = Conv_Block(32, 64 , 3, int(4/enlarge))

        self.gru = nn.GRU(64*enlarge, 64*enlarge, 2, dropout=self.dropout,batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128*enlarge, class_num)

    def detection(self, x):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     x: [batch_size, time_steps, num_freq]
        # Return:
        #     frame_wise_prob: [batch_size, time_steps, class_num]
        ##############################
        x = x.permute(0, 2, 1)
        x = self.batch_norm(x)
        x = x.unsqueeze(1).permute(0,1,3,2)
        x = self.conv3(self.conv2(self.conv1(x)))
        x = x.permute(0, 2, 1, 3)
        x = x.flatten(start_dim=2)
        x = self.gru(x)[0]
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

    def forward(self, x):
        frame_wise_prob = self.detection(x)  # B, T, ncls
        clip_prob = linear_softmax_pooling(frame_wise_prob)  # B, ncls
        '''(samples_num, feature_maps)'''
        return {
            'clip_probs': clip_prob,
            'time_probs': frame_wise_prob
        }


class Crnn(nn.Module):
    def __init__(self, num_freq, class_num, dropout, enlarge: int =1):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     num_freq: int, mel frequency bins
        #     class_num: int, the number of output classes
        ##############################
        super().__init__()
        self.num_freq = num_freq
        self.class_num = class_num
        self.dropout = dropout
        self.batch_norm = nn.BatchNorm1d(64)
        self.conv1 = Conv_Block(1, 16 , 3 , 4)
        self.conv2 = Conv_Block(16, 32 , 3, 4)
        self.conv3 = Conv_Block(32, 64 , 3, int(4/enlarge))

        self.lstm = nn.LSTM(64*enlarge, 64*enlarge, 2, dropout=self.dropout,batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128*enlarge, class_num)

    def detection(self, x):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     x: [batch_size, time_steps, num_freq]
        # Return:
        #     frame_wise_prob: [batch_size, time_steps, class_num]
        ##############################
        x = x.permute(0, 2, 1)
        x = self.batch_norm(x)
        x = x.unsqueeze(1).permute(0,1,3,2)
        x = self.conv3(self.conv2(self.conv1(x)))
        x = x.permute(0, 2, 1, 3)
        x = x.flatten(start_dim=2)
        x = self.lstm(x)[0]
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

    def forward(self, x):
        frame_wise_prob = self.detection(x)  # B, T, ncls
        clip_prob = linear_softmax_pooling(frame_wise_prob)  # B, ncls
        '''(samples_num, feature_maps)'''
        return {
            'clip_probs': clip_prob,
            'time_probs': frame_wise_prob
        }





class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel ,kernel_size,pooling_size):
        super().__init__()
        self.pooling_size = pooling_size
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=kernel_size // 2)
        self.batch_norm = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        if self.pooling_size>1:
            self.pooling = nn.MaxPool2d((1,pooling_size))

    def forward(self, x):
        x = self.relu(self.batch_norm(self.conv(x)))
        if self.pooling_size > 1:
            x = self.pooling(x)
        return x

class Res_Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, pooling_size):
        super().__init__()

        self.flag = False
        if in_channel != out_channel:
            self.flag = True
            self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channel)
        )
        self.relu = nn.ReLU()

        self.pooling = nn.MaxPool2d((1, pooling_size))

    def forward(self, x):

        tmp = self.conv(x) if self.flag == True else x
        x = self.block(x)
        x += tmp
        x = self.relu(x)
        x = self.pooling(x)
        return x


class Crnn_res(nn.Module):
    def __init__(self, num_freq, class_num, dropout, enlarge: int =1):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     num_freq: int, mel frequency bins
        #     class_num: int, the number of output classes
        ##############################
        super().__init__()
        self.num_freq = num_freq
        self.class_num = class_num
        self.dropout = dropout
        self.batch_norm = nn.BatchNorm1d(64)
        self.conv1 = Res_Block(1, 16 , 3 , 4)
        self.conv2 = Res_Block(16, 32 , 3, 4)
        self.conv3 = Res_Block(32, 64 , 3, int(4/enlarge))

        self.gru = nn.GRU(64*enlarge, 64*enlarge, 2, dropout=self.dropout,batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128*enlarge, class_num)

    def detection(self, x):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     x: [batch_size, time_steps, num_freq]
        # Return:
        #     frame_wise_prob: [batch_size, time_steps, class_num]
        ##############################
        x = x.permute(0, 2, 1)
        x = self.batch_norm(x)
        x = x.unsqueeze(1).permute(0,1,3,2)
        x = self.conv1(x)
        x = self.conv3(self.conv2(x))
        x = x.permute(0, 2, 1, 3)
        x = x.flatten(start_dim=2)
        x = self.gru(x)[0]
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

    def forward(self, x):
        frame_wise_prob = self.detection(x)  # B, T, ncls
        clip_prob = linear_softmax_pooling(frame_wise_prob)  # B, ncls
        '''(samples_num, feature_maps)'''
        return {
            'clip_probs': clip_prob,
            'time_probs': frame_wise_prob
        }