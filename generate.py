import argparse, librosa
import numpy as np

import soundfile as sf
import torch
import torch.nn as nn
from torch.autograd import Variable

from model import Tacotron as Tacotron
from collate_fn import collate_class
from dataset import *
from util import *
from stft import STFT
import os

parser = argparse.ArgumentParser(description='training script')
# data load
parser.add_argument('--data', type=str, default='vctk', help='vctk')
parser.add_argument('--batch_size', type=int, default=2, help='batch size')
parser.add_argument('--text_limit', type=int, default=200, help='maximum length of text to include in training set')
parser.add_argument('--wave_limit', type=int, default=1220, help='maximum length of spectrogram to include in training set')
parser.add_argument('--trunc_size', type=int, default=610, help='used for truncated-BPTT when memory is not enough.')
parser.add_argument('--shuffle_data', type=int, default=0, help='whether to shuffle data loader')
parser.add_argument('--batch_idx', type=int, default=0, help='n-th batch of the dataset')
parser.add_argument('--load_queue_size', type=int, default=1, help='maximum number of batches to load on the memory')
parser.add_argument('--n_workers', type=int, default=0, help='number of workers used in data loader')
# generation option
parser.add_argument('--out_dir', type=str, default='generated', help='')
parser.add_argument('--init_from', type=str, default='./pretrained_model.pt', help='load parameters from...')
parser.add_argument('--caption', type=str, default='', help='text to generate speech')
parser.add_argument('--speaker_id', type=str, default='0', help='speaker id to generate speech, seperate by comma for mixing id')
parser.add_argument('--teacher_forcing_ratio', type=float, default=0, help='value between 0~1, use this for scheduled sampling')
parser.add_argument('--use_txt', type=float, default=0, help='value between 0~1, use this for scheduled sampling')
# audio related option
parser.add_argument('--n_fft', type=int, default=2048, help='fft bin size')
parser.add_argument('--sample_rate', type=int, default=16000, help='sampling rate')
parser.add_argument('--frame_len_inMS', type=int, default=50, help='used to determine window size of fft')
parser.add_argument('--frame_shift_inMS', type=int, default=12.5, help='used to determine stride in sfft')
parser.add_argument('--num_recon_iters', type=int, default=50, help='# of iteration in griffin-lim recon')
# misc
parser.add_argument('--gpu', type=int, nargs='+', help='index of gpu machines to run')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--att_type', type=str, default='Bahdanau', help='attention type: Bahdanau, Monotonic')
parser.add_argument('--postfix', type=str, default='', help='postfix of savename')
new_args = vars(parser.parse_args())

# load and override some arguments
checkpoint = torch.load(new_args['init_from'], map_location=lambda storage, loc: storage)
args = checkpoint['args']
for i in new_args:
    args.__dict__[i] = new_args[i]

torch.manual_seed(args.seed)

if args.gpu is None:
    args.use_gpu = False
    args.gpu = []
else:
    args.use_gpu = True
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu[0])

model = Tacotron(args)
if args.init_from:
    model.load_state_dict(checkpoint['state_dict'])
    model.reset_decoder_states()
    print('loaded checkpoint %s' % (args.init_from))

stft = STFT(filter_length=args.n_fft)
model = model.eval()
if args.use_gpu:
    model = model.cuda()
    stft = stft.cuda()

def main():
    db = TTSDataset()
    collate = collate_class(use_txt=args.use_txt)
    loader = torch.utils.data.DataLoader(db, batch_size=1, shuffle=False, collate_fn=collate.fn, drop_last=True)
    model_name = args.init_from.split('/')[-1][:-3]

    for ii, sample in enumerate(loader):
        if args.caption:
            print('[*] Caption detected: {}'.format(args.caption))
            txt = args.caption
            txt = decompose_hangul(txt)
            txt = list(filter(None, [db.char2onehot(xx) for xx in txt]))
            txt = torch.tensor(txt).unsqueeze(0)
            txt_len[0] = txt.shape[-1]
            sample['txt'] = txt
            sample['txt_len'] = txt_len

        for k, v in sample.items():
            if k in ['filename', 'contents_domain']:
                continue
            if args.use_gpu:
                sample[k] = Variable(v, requires_grad=False).cuda()
            else:
                sample[k] = Variable(v, requires_grad=False)

        wave, attentions, style_vec, pred_mel, context_vec = generate(sample)
        contents_filename = os.path.basename(sample['filename'][0]['input'])[:-4]
        style_filename = os.path.basename(sample['filename'][0]['ref'])[:-4]
        target_filename = os.path.basename(sample['filename'][0]['target'])[:-4]
        contents_domain = sample['contents_domain']

        if args.caption:
            contents_filename = txt[:10]
        outpath1 = '%s/%s_%s_%s_%s_%s.wav' % (args.out_dir, model_name, contents_filename, style_filename, target_filename, contents_domain)
        librosa.output.write_wav(outpath1, wave, 16000)
        outpath2 = '%s/%s_%s_%s_%s_%s.png' % (args.out_dir, model_name, contents_filename, style_filename, target_filename, contents_domain)
        saveAttention(None, attentions, outpath2)
        outpath3 = '%s/%s_%s_%s_%s_%s.pt' % (args.out_dir, model_name, contents_filename, style_filename, target_filename, contents_domain)
        torch.save({'style_vec': style_vec, 'mel': pred_mel.detach().cpu().numpy(), 'context_vec': context_vec, 'att': attentions}, outpath3)
        print(outpath2)
        
def generate(sample):
    model.reset_decoder_states()
    model.mask_decoder_states()
    pred_mel, pred_lin, _ = model(**sample)
    attentions = torch.cat(model.attn_weights, dim=-1)
    style_vec = model.style_vec.detach().cpu().numpy()
    context_vec = model.context_vec

    window_len = int(np.ceil(args.frame_len_inMS * args.sample_rate / 1000))
    hop_length = int(np.ceil(args.frame_shift_inMS * args.sample_rate / 1000))

    # write to file
    wave = spectrogram2wav_gpu(
        pred_lin.data,
        n_fft=args.n_fft,
        win_length=window_len,
        hop_length=hop_length,
        num_iters=args.num_recon_iters,
        stft=stft
    )
    wave = wave.squeeze().cpu().numpy()


    return wave, attentions[0], style_vec, pred_mel, context_vec

def spectrogram2wav_gpu(magnitudes, n_fft, win_length, hop_length, num_iters, stft):
    '''
    spectrogram: [t, f], i.e. [t, nfft // 2 + 1]
    '''
    min_level_db = -100
    ref_level_db = 20

    # denormalize
    magnitudes = magnitudes.permute(0, 2, 1)
    magnitudes = (torch.clamp(magnitudes, 0, 1) * - min_level_db) + min_level_db
    magnitudes = magnitudes + ref_level_db

    # Convert back to linear
    magnitudes = torch.pow(10.0, magnitudes * 0.05)
    magnitudes = magnitudes ** 1.5

    angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.shape)))
    angles = angles.astype(np.float32)
    angles = Variable(torch.from_numpy(angles))
    angles = angles.to(magnitudes.device)
    signal = stft.inverse(magnitudes, angles).squeeze(1)

    for i in range(num_iters):
        _, angles = stft.transform(signal)
        signal = stft.inverse(magnitudes, angles).squeeze(1)
    return signal

def saveAttention(input_sentence, attentions, outpath):
    # Set up figure with colorbar
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    fig, ax = plt.subplots()
    cax = ax.matshow(attentions.cpu().numpy(), aspect='auto', origin='upper',cmap='gray')
    # fig.colorbar(cax)
    plt.ylabel('Encoder timestep', fontsize=18)
    plt.xlabel('Decoder timestep', fontsize=18)

    if input_sentence:
        plt.ylabel('Encoder timestep', fontsize=18)

    plt.tight_layout()
    plt.savefig(outpath)
    plt.close('all')


def spectrogram2wav(spectrogram, n_fft, win_length, hop_length, num_iters):
    '''
    spectrogram: [t, f], i.e. [t, nfft // 2 + 1]
    '''
    min_level_db = -100
    ref_level_db = 20

    spec = spectrogram.T
    # denormalize
    spec = (np.clip(spec, 0, 1) * - min_level_db) + min_level_db
    spec = spec + ref_level_db

    # Convert back to linear
    spec = np.power(10.0, spec * 0.05)

    return _griffin_lim(spec ** 1.5, n_fft, win_length, hop_length, num_iters)  # Reconstruct phase


def _griffin_lim(S, n_fft, win_length, hop_length, num_iters):
    # angles = np.exp(2j * np.pi * np.ones(S.shape))
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    for i in range(num_iters):
        if i > 0:
            angles = np.exp(1j * np.angle(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)))
        y = librosa.istft(S_complex * angles, hop_length=hop_length, win_length=win_length)
    return y

if __name__ == '__main__':
    main()
