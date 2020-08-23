import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, emb_type='last', *args, **kargs):
        super().__init__()
        self.__dict__.update(locals())
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
    def forward(self, x, x_len):
        x_len, x_len_arg = x_len.sort(descending=True) 
        _, x_len_back = (x_len_arg).sort(descending=False)

        x = x[x_len_arg].contiguous()
        
        x_pack = rnn.pack_padded_sequence(x, x_len, batch_first=True)
        y_pack, (h_pack, _) = self.lstm(x_pack)
        y, _ = rnn.pad_packed_sequence(y_pack, batch_first=True)
        y = self.integrate(y, x_len)
        
        if y.ndim == 2:
            y = y[x_len_back].unsqueeze(1)
        return y

    def integrate(self, y, y_len):
        if self.emb_type=='all':
            y = y.sum(dim=1)
            y = y.div(y_len.unsqueeze(1).float())
        elif self.emb_type=='firstlast':
            masks = (y_len-1).view(-1, 1).expand(len(y_len), y.shape[2]).unsqueeze(1)
            y_last = outputs.gather(1, masks).squeeze(1)[:, self.hidden_size:]
            y = (y[:, 0, :self.hidden_size] + y_last) / 2
        elif self.emb_type=='first':
            y = y[:, 0, :self.hidden_size]
        elif self.emb_type=='last':
            masks = (y_len-1).view(-1, 1).expand(len(y_len), y.shape[2]).unsqueeze(1)
            y = y.gather(1, masks).squeeze(1)[:, self.hidden_size:]
        elif self.emb_type=='raw':
            y = y
        return y
