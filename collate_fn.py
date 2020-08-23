import torch
import torch.utils.data as data
import random
import numpy as np

class collate_class:
    def __init__(self, use_txt):
        self.use_txt = use_txt

    def fn(self, data):
        if random.random() < self.use_txt:
            contents_domain = 'txt'
        else:
            contents_domain = 'input_mel'

        n_batch = len(data)
        data.sort(key=lambda x: len(x[contents_domain]), reverse=True)

        input_mel_len = torch.tensor([len(x['input_mel']) for x in data])
        max_input_mel_len = max(input_mel_len)
        target_mel_len = torch.tensor([len(x['target_mel']) for x in data])
        max_target_mel_len = max(target_mel_len)
        txt_len = torch.tensor([len(x['txt']) for x in data])
        max_txt_len = max(txt_len)
        ref_mel_len = torch.tensor([len(x['ref_mel']) for x in data])
        max_ref_mel_len = max(ref_mel_len)
        max_lin_len = max([len(x['lin']) for x in data])

        txt = torch.zeros(n_batch, max_txt_len).long()
        target_mel = torch.zeros(n_batch, max_target_mel_len, data[0]['target_mel'].shape[-1])
        ref_mel = torch.zeros(n_batch, max_ref_mel_len, data[0]['ref_mel'].shape[-1])
        input_mel = torch.zeros(n_batch, max_input_mel_len, data[0]['input_mel'].shape[-1])
        lin = torch.zeros(n_batch, max_lin_len, data[0]['lin'].shape[-1])

        gender = torch.zeros(n_batch).long()
        age = torch.zeros(n_batch).long()
        emotion = torch.zeros(n_batch).long()
        spkemb = torch.zeros((n_batch, 256))
        filename = []

        for ii, item in enumerate(data):
            ref_mel[ii, :len(item['ref_mel'])] = torch.tensor(item['ref_mel'])
            target_mel[ii, :len(item['target_mel'])] = torch.tensor(item['target_mel'])
            input_mel[ii, :len(item['input_mel'])] = torch.tensor(item['input_mel'])
            lin[ii, :len(item['lin'])] = torch.tensor(item['lin'])
            txt[ii, :len(item['txt'])] = torch.tensor(item['txt']).long()

            gender[ii]  = item['style']['gender']
            age[ii]     = item['style']['age']
            emotion[ii] = item['style']['emotion']
            if 'speaker' in item['style'].keys():
                spkemb[ii] = torch.tensor(item['style']['speaker'])
            filename.append(item['filename'])

        out_list = ['target_mel', 'lin', 'txt', 'input_mel', 'ref_mel',
                    'target_mel_len', 'txt_len', 'input_mel_len', 'ref_mel_len',
                    'gender', 'age', 'emotion', 'spkemb', 'filename',
                    'contents_domain']

        if not contents_domain == 'txt':
            out_list = [xx for xx in out_list if xx not in ['txt', 'txt_len']]
        elif not contents_domain == 'input_mel':
            out_list = [xx for xx in out_list if xx not in ['input_mel', 'input_mel_len']]

        return_dict = {k:v for k, v in locals().items() if k in out_list}

        assert len(out_list) == len(return_dict)

        return return_dict 
