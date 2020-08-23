import torch
import torch.utils.data as data
from glob import glob
from os.path import join, basename, exists
import numpy as np
import pickle as pkl
from random import random
np.random.seed(123)

class TTSDataset(data.Dataset):
    def __init__(self, which_set='train', datapath='./samples'):
        # Load vocabulary
        vocab_path = datapath + '/vocab_dict.pkl'
        self.vocab_dict = pkl.load(open(vocab_path, 'rb'))
        self.vocab_size = len(self.vocab_dict)

        # Filelist 
        self.txtlist = np.sort(glob(datapath+'/*.txt'))
        self.mellist = np.sort(glob(datapath+'/*.mel'))

        sent_no = lambda x: int(basename(x).split('_')[1][:5])
        
        self.gen_lu = {'female': 0, 'male': 1}
        self.age_lu = {'age20': 0, 'age30': 1, 'age40': 2}
        self.emo_lu = {'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3, 'sur': 4, 'fea': 5, 'dis': 6}
        assert len(self.txtlist)==len(self.mellist), \
                'mellist({}) and txtlist({}) has different length'.format(len(self.mellist), len(self.txtlist))

        self.char2onehot = lambda x : self.vocab_dict[x] if x in self.vocab_dict.keys() else None 

    def __len__(self):
        return len(self.txtlist)

    def __getitem__(self, idx):
        '''
        Be sure that 
            contents(input_mel) == contents(target_mel) == txt
            style(ref_mel) == style(target_mel)
        '''
        # Text read
        with open(self.txtlist[idx], 'r') as f:
            txt = f.readline()
        txt_feat = list(filter(None, [self.char2onehot(xx) for xx in txt]))

        # load mel/lin of x_o
        mellin = pkl.load(open(self.mellist[idx], 'rb'))
        mel = mellin['mel']
        lin = mellin['lin']

        # Get path of x_s
        mel_emo = basename(self.mellist[idx])[:3]
        emo_set = sorted(self.emo_lu.keys())
        emo_set.remove(mel_emo)
        emo_set = np.random.permutation(emo_set)
        for input_emo in emo_set:
            input_path = self.mellist[idx].replace(mel_emo, input_emo)
            if exists(input_path):
                break

        # Get path of x_c
        while True:
            sent_no = '{:05d}'.format(np.random.randint(3000))
            ref_path = self.mellist[idx]
            ref_path = ref_path.replace(ref_path[-9:-4], sent_no)
            if exists(ref_path):
                break

        input_mel = pkl.load(open(input_path, 'rb'))['mel']
        ref_mel = pkl.load(open(ref_path, 'rb'))['mel']
        style = self.getstyle(self.txtlist[idx])

        return {'txt': np.asarray(txt_feat), 
                'style': style, 
                'lin': np.asarray(lin), 
                'target_mel': np.asarray(mel),
                'ref_mel': np.asarray(ref_mel),
                'input_mel': np.asarray(input_mel),
                'filename': {'target':self.mellist[idx], 'ref':ref_path, 'input':input_path}
                }

    def getstyle(self, filename):
        filename = basename(filename)
        gender = self.gen_lu['male']
        age = self.age_lu['age30']
        emotion = self.emo_lu[filename[:3]]
        return {'age': age, 'gender': gender,'emotion': emotion}

    def get_vocab_size(self):
        return self.vocab_size
