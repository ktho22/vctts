from multiprocessing import Process, Queue, Pool
import os, glob, re, librosa, argparse, torch, pickle, multiprocessing
import pickle as pkl
from os.path import *
import numpy as np
from util import decompose_hangul

def preprocess_text(txtlist, dir_bin, lang):
    vocab_set = set()
    text_file_list = []
    dirts = ['`', '#', '@', '\|', '\)', '"', '\ufeff']
    for fname in txtlist:
        gname = join(dir_bin, basename(fname))
        text_file_list.append(gname)
        with open(fname, 'r') as f, open(gname, 'w') as g:
            line = f.readline()
            line = ''.join([xx for xx in line if xx not in dirts])
            if lang == 'korean':
                line = decompose_hangul(line)
            g.write(line)
            vocab_set = vocab_set.union(set(line))

    vocab_dict = dict(zip(vocab_set, range(len(vocab_set))))
    print('Final vocab: ', vocab_dict)
    with open(dir_bin + '/../vocab_dict.pkl', 'wb') as g:
        pkl.dump(vocab_dict, g)
    return text_file_list

def preprocess_spec(wavlist, dir_bin, sample_rate=16000, nfft=2048, type_filter='both'):
    '''
        Preprocessing for mel- and lin- spectrogram
        Args: wavlist, dir_bin
        Return: speclist [(wavname, mel_length, mel_bin_length, lin_length, lin_bin_length)]
    '''
    print('Start writing %s spectrogram binary files' % type_filter)
    offset_m, offset_l = 0, 0
    p = Pool()
    map_return = p.map_async(inner_process_spec, wavlist)
    p.close(); p.join()
    return map_return.get()


def inner_process_spec(wavname, sample_rate=16000, isMono=True, type_filter='both', frame_len_inMS=50, 
        frame_shift_inMS=12.5, n_fft=2048, ref_level_db=20, min_level_db=-100, mel_dim=80, wav_limit=1220):

    try:
        write_path_mel = join(dir_bin, basename(wavname)[:-4] + '.mel')

        audio,_ = librosa.load(wavname, sr=sample_rate, mono=isMono)
        mel_basis = librosa.filters.mel(sample_rate, n_fft, n_mels=mel_dim)

        # params for stft
        window_len = int(np.ceil(frame_len_inMS * sample_rate / 1000))
        hop_length = int(np.ceil(frame_shift_inMS * sample_rate / 1000))

        D = librosa.stft(audio, n_fft=n_fft, win_length=window_len, window='hann', hop_length=hop_length)
        spec = np.abs(D)

        # mel-scale spectrogram generation
        spec_mel = np.dot(mel_basis, spec)
        spec_mel = 20 * np.log10(np.maximum(1e-5, spec_mel))
        # linear spectrogram generation
        spec_lin = 20 * np.log10(np.maximum(1e-5, spec)) - ref_level_db

        # normalize
        spec_mel = np.clip(-(spec_mel - min_level_db) / min_level_db, 0, 1)
        spec_mel = spec_mel.T
        spec_lin = np.clip(-(spec_lin - min_level_db) / min_level_db, 0, 1)
        spec_lin = spec_lin.T
        
        if len(spec_mel) > wav_limit:
            return None

        with open(write_path_mel, 'wb') as w_mel:
            bin_spec_mel = pickle.dumps({'mel':spec_mel,'lin':spec_lin}, protocol=pickle.HIGHEST_PROTOCOL)
            w_mel.write(bin_spec_mel)
    except Exception as e:
        print(e)

    return wavname 

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Preprocessing dataset')
    parser.add_argument('--wav_dir', type=str, help='input wav folder')
    parser.add_argument('--txt_dir', type=str, help='input txt folder')
    parser.add_argument('--bin_dir', type=str, help='output bin folder')
    parser.add_argument('--lang', type=str, default='korean')

    args = parser.parse_args()
    print('[*] Preprocessing Database')

    dir_spec = args.wav_dir
    dir_text = args.txt_dir
    dir_bin = args.bin_dir

    txtlist = sorted(glob.glob(join(dir_text, '*.txt')))
    wavlist = sorted(glob.glob(join(dir_spec, '*.wav')))
    assert len(txtlist)==len(wavlist), "number of txt files and wav files should be same"
    print(len(txtlist))

    txtlist = []
    speclist = preprocess_spec(wavlist, dir_bin) 
    print(len(speclist), len(wavlist))
    for specfile in speclist:
        if specfile == None:
            continue
        txtlist.append(os.path.join(dir_text, splitext(basename(specfile))[0])+'.txt')
    textlist = preprocess_text(txtlist, dir_bin, args.lang)

    for p in multiprocessing.active_children():
        p.terminate()
