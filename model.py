from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import random
from encoder import EncoderRNN

class Tacotron(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.use_gpu = args.use_gpu
        self.r_factor = args.r_factor
        self.dec_out_size = args.dec_out_size

        self.teacher_forcing_ratio = args.teacher_forcing_ratio
        self.attn_weights = []

        # text encoder 
        self.encoder = Encoder(args.vocab_size, args.charvec_dim, args.hidden_size, \
                args.num_filters)
        self.linear_enc = nn.Linear(2 * args.hidden_size, 2 * args.hidden_size, bias=False)

        self.contents_enc = EncoderRNN(80, 128, 2, emb_type='raw')

        # decoder
        self.decoder = AttnDecoderRNN(args.hidden_size, args.dec_out_size, args.style_embed_size, \
                args.r_factor)
        self.post_processor = PostProcessor(args.hidden_size, args.dec_out_size, args.post_out_size, args.num_filters // 2)

        # style
        self.style_enc = EncoderRNN(80, 32, 2)

        self.gender_embed = nn.Embedding(args.gender_num, args.style_embed_size)
        self.emotion_embed = nn.Embedding(args.emotion_num, args.style_embed_size)
        self.age_embed = nn.Embedding(args.age_num, args.style_embed_size)
        self.spk_embed_1 = nn.Linear(256, 10)
        self.spk_embed_2 = nn.Linear(10, 32)

        self.reset_decoder_states()


    def forward(self, target_mel, target_mel_len, 
            txt=None, txt_len=None,
            ref_mel=None, ref_mel_len=None,
            input_mel=None, input_mel_len=None,
            gender=None, age=None, emotion=None, spkemb=None, epoch=0, *args, **kwargs):
        N = target_mel.size(0) # Batch size
        r = self.r_factor
        wav_len_max = int(target_mel.shape[1])
        dec_len_max = wav_len_max // r

        if self.use_gpu:
            output_mel = torch.zeros([N, wav_len_max, target_mel.shape[-1]]).cuda()
        else:
            output_mel = torch.zeros([N, wav_len_max, target_mel.shape[-1]])

        if torch.is_tensor(input_mel):
            contents_len = input_mel_len
            enc_output = self.contents_enc(input_mel, input_mel_len)
            in_attW_enc = rnn.pack_padded_sequence(enc_output, input_mel_len, True)
        elif torch.is_tensor(txt):
            contents_len = txt_len
            enc_output = self.encoder(txt, txt_len)
            in_attW_enc = rnn.pack_padded_sequence(enc_output, txt_len, True)
        in_attW_enc = self.linear_enc(in_attW_enc.data)

        # style enc
        style_vec = self.style_enc(ref_mel, ref_mel_len)
        self.style_vec = style_vec
        self.context_vec = []

        # Unrolling RNN
        for di in range(dec_len_max + 1):
            self.prev_dec_output, self.prev_context = \
                self.decoder(enc_output, in_attW_enc, self.prev_dec_output, style_vec, contents_len, self.prev_context)
            self.context_vec.append(self.prev_context.detach().cpu().numpy())

            if  di < dec_len_max:
                output_mel[:, di*r:di*r+r, :] = self.prev_dec_output

                if random.random() < self.teacher_forcing_ratio:
                    self.prev_dec_output = target_mel[:, di * r - 1]
                else:
                    self.prev_dec_output = self.prev_dec_output[:, -1]

            # If `mel` has residual length after division by `r_factor`
            else:
                res_length = wav_len_max % r
                if res_length == 0:
                    break
                output_mel[:, di*r:, :] = self.prev_dec_output[:, :res_length, :]

            self.attn_weights.append(self.decoder.attn_weights.data)

        output_linear = self.post_processor(output_mel)

        return output_mel, output_linear, self.attn_weights

    def reset_decoder_states(self):
        self.decoder.reset_states()
        self.prev_dec_output = None
        self.prev_context = None
        self.attn_weights = []

    def mask_decoder_states(self, len_mask=None):
        # wrapper
        self.decoder.mask_states(len_mask)

        if self.prev_dec_output is not None:
            if len_mask is None:
                self.prev_dec_output = Variable(self.prev_dec_output.data, requires_grad=False)
                self.prev_context = Variable(self.prev_context.data, requires_grad=False)
            else:
                self.prev_dec_output = Variable(torch.index_select(self.prev_dec_output.data, 0, len_mask), requires_grad=False)
                self.prev_context = Variable(torch.index_select(self.prev_context.data, 0, len_mask), requires_grad=False)


class Encoder(nn.Module):
    """ input[0]: NxT sized Tensor
        input[1]: B sized lengths Tensor
        output: NxTxH sized Tensor
    """
    def __init__(self, vocab_size, charvec_dim, hidden_size, num_filters, dropout_p=0.5):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, charvec_dim)
        self.prenet = nn.Sequential(
            nn.Linear(charvec_dim, 2 * hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )
        self.CBHG = CBHG(hidden_size, hidden_size, hidden_size, hidden_size, hidden_size, num_filters, True)

    def forward(self, input, lengths):
        output = self.prenet(self.embedding(input))
        output = self.CBHG(output, lengths)
        return output


class AttnDecoderRNN(nn.Module):
    """ input_enc: Output from encoder (NxTx2H)
        input_attW_enc: masked-linear transformed input_enc
        input_dec: Output from previous-step decoder (NxO_dec)
        style_vec: Speaker embedding (Nx1xS)
        lengths: N sized Tensor
        output: N x r x H sized Tensor
    """
    def __init__(self, hidden_size, output_size, style_embed_size, r_factor=2, dropout_p=0.5):
        super().__init__()
        self.r_factor = r_factor
        self.O_dec = output_size

        self.prenet = nn.Sequential(
            nn.Linear(output_size, 2 * hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )
        self.linear_dec = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.gru_att = nn.GRU(3 * hidden_size + style_embed_size, 2 * hidden_size, batch_first=True)
        self.attn = nn.Linear(2 * hidden_size, 1)

        self.short_cut = nn.Linear(4 * hidden_size + style_embed_size, 2 * hidden_size)
        self.gru_dec1 = nn.GRU(4 * hidden_size + style_embed_size, 2 * hidden_size, num_layers=1, batch_first=True)
        self.gru_dec2 = nn.GRU(2 * hidden_size, 2 * hidden_size, num_layers=1, batch_first=True)

        self.out = nn.Linear(2 * hidden_size, r_factor * output_size)

        self.reset_states()

    def forward(self, input_enc, input_attW_enc, input_dec, style_vec, lengths_enc, input_context=None):
        N, T_enc = input_enc.size(0), max(lengths_enc)

        if input_dec is None:
            input_dec = Variable(input_enc.data.new().resize_(N, self.O_dec).zero_(), requires_grad=False)
        out_prenet = self.prenet(input_dec).unsqueeze(1)

        if input_context is None:
            input_context = Variable(input_enc.data.new().resize_(N, 1, out_prenet.size(2)*2).zero_(), requires_grad=False)
        in_gru_att = torch.cat([out_prenet, input_context, style_vec], dim=2)

        out_att, self.hidden_att = self.gru_att(in_gru_att, self.hidden_att)
        in_attW_dec = self.linear_dec(out_att).expand_as(input_enc)
        in_attW_dec = rnn.pack_padded_sequence(in_attW_dec, lengths_enc, True)      

        e = torch.add(input_attW_enc, in_attW_dec.data).tanh()                      
        e = self.attn(e)                                                            

        # Bahdanau attention
        self.attn_weights = rnn.PackedSequence(e.exp(), in_attW_dec.batch_sizes)
        self.attn_weights, _ = rnn.pad_packed_sequence(self.attn_weights, True)
        self.attn_weights = F.normalize(self.attn_weights, 1, 1)

        attn_applied = torch.bmm(self.attn_weights.transpose(1, 2), input_enc)

        out_dec = torch.cat((attn_applied, out_att, style_vec), 2)
        residual = self.short_cut(out_dec)

        out_dec, self.hidden_dec1 = self.gru_dec1(out_dec, self.hidden_dec1)
        residual = residual + out_dec

        out_dec, self.hidden_dec2 = self.gru_dec2(residual, self.hidden_dec2)
        residual = residual + out_dec

        output = self.out(residual).view(N, self.r_factor, -1)

        return output, attn_applied

    def reset_states(self):
        self.hidden_att = None
        self.hidden_dec1 = None
        self.hidden_dec2 = None
        self.attn_weights = None

    def mask_states(self, len_mask):
        if self.hidden_att is not None:
            if len_mask is None:
                self.hidden_att = Variable(self.hidden_att.data, requires_grad=False)
                self.hidden_dec1 = Variable(self.hidden_dec1.data, requires_grad=False)
                self.hidden_dec2 = Variable(self.hidden_dec2.data, requires_grad=False)
            else:
                self.hidden_att  = Variable(torch.index_select(self.hidden_att.data, 1, len_mask), requires_grad=False)
                self.hidden_dec1 = Variable(torch.index_select(self.hidden_dec1.data, 1, len_mask), requires_grad=False)
                self.hidden_dec2 = Variable(torch.index_select(self.hidden_dec2.data, 1, len_mask), requires_grad=False)

class PostProcessor(nn.Module):
    """ input: N x T x O_dec
        output: N x T x O_post
    """
    def __init__(self, hidden_size, dec_out_size, post_out_size, num_filters):
        super().__init__()
        self.CBHG = CBHG(dec_out_size, hidden_size, 2 * hidden_size, hidden_size, hidden_size, num_filters, True)
        self.projection = nn.Linear(2 * hidden_size, post_out_size)

    def forward(self, input, lengths=None):
        if lengths is None:
            N, T = input.size(0), input.size(1)
            lengths = [T for _ in range(N)]
            output = self.CBHG(input, lengths).contiguous()
            output = self.projection(output)
        else:
            output = self.CBHG(input, lengths)
            output = rnn.pack_padded_sequence(output, lengths, True)
            output = rnn.PackedSequence(self.projection(output.data), output.batch_sizes)
            output, _ = rnn.pad_packed_sequence(output, True)
        return output


class CBHG(nn.Module):
    """ input: NxTxinput_dim sized Tensor
        output: NxTx2gru_dim sized Tensor
    """
    def __init__(self, input_dim, conv_bank_dim, conv_dim1, conv_dim2, gru_dim, num_filters, is_masked):
        super().__init__()
        self.num_filters = num_filters

        bank_out_dim = num_filters * conv_bank_dim
        self.conv_bank = nn.ModuleList()
        for i in range(num_filters):
            self.conv_bank.append(nn.Conv1d(input_dim, conv_bank_dim, i + 1, stride=1, padding=int(np.ceil(i / 2))))

        # define batch normalization layer, we use BN1D since the sequence length is not fixed
        self.bn_list = nn.ModuleList()
        self.bn_list.append(nn.BatchNorm1d(bank_out_dim))
        self.bn_list.append(nn.BatchNorm1d(conv_dim1))
        self.bn_list.append(nn.BatchNorm1d(conv_dim2))

        self.conv1 = nn.Conv1d(bank_out_dim, conv_dim1, 3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(conv_dim1, conv_dim2, 3, stride=1, padding=1)

        if input_dim != conv_dim2:
            self.residual_proj = nn.Linear(input_dim, conv_dim2)

        self.highway = Highway(conv_dim2, 4)
        self.rnn_residual = nn.Linear(conv_dim2, 2*conv_dim2)
        self.BGRU = nn.GRU(input_size=conv_dim2, hidden_size=gru_dim, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, input, lengths):
        N, T = input.size(0), input.size(1)

        conv_bank_out = []
        for i in range(self.num_filters):
            tmp_input = input.transpose(1, 2)
            if i % 2 == 0:
                tmp_input = tmp_input.unsqueeze(-1)
                tmp_input = F.pad(tmp_input, (0,0,0,1)).squeeze(-1)
            conv_bank_out.append(self.conv_bank[i](tmp_input))

        residual = torch.cat(conv_bank_out, dim=1)
        residual = F.relu(self.bn_list[0](residual))
        residual = F.max_pool1d(residual, 2, stride=1)
        residual = self.conv1(residual)
        residual = F.relu(self.bn_list[1](residual))
        residual = self.conv2(residual)
        residual = self.bn_list[2](residual).transpose(1,2)

        rnn_input = input
        if rnn_input.size() != residual.size():
            rnn_input = self.residual_proj(rnn_input)
        rnn_input = rnn_input + residual
        rnn_input = self.highway(rnn_input)

        output = rnn.pack_padded_sequence(rnn_input, lengths, True)
        output, _ = self.BGRU(output)
        output, _ = rnn.pad_packed_sequence(output, True)

        rnn_residual = self.rnn_residual(rnn_input)
        output = rnn_residual + output
        return output


class Highway(nn.Module):
    def __init__(self, size, num_layers, f=F.relu):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.f = f

    def forward(self, x):
        """ input: NxH sized Tensor
            output: NxH sized Tensor
        """
        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear
        return x
