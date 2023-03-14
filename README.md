This is the code of the paper **Emotional Voice Conversion using Multitask Learning with Text-to-speech**, ICASSP 2020 [[link]](https://arxiv.org/abs/1911.06149)



## Prerequisite

Install required packages 

```shell
pip3 install -r requirements.txt
```



## Inference

Few samples and pretraiend model for VC are provided, so you can try with below command.

Samples contain 20 types of sentences and 7 emotions, 140 utterances in total.


~[[model download]](http://gofile.me/4B76q/yobaWLDtb)~

~[[samples download]](http://gofile.me/4B76q/RkQxQuKvY)~

Model/samples download links are expired. 

```shell
python3 generate.py --init_from <model_path> --gpu <gpu_id> --out_dir <out_dir>
```

Below is an example of generated wav.

It means the model takes contents of `(fear, 20th contents)` and style of `(anger, 2nd contents)` to make `(anger, 20th contents)`.

```shell
pretrained_model_fea_00020_ang_00002_ang_00020_input_mel.wav
```


## Training

You can train your own dataset, by changing contents of `dataset.py`

```shell
# remove silence within wav files
python3 trimmer.py --in_dir <in_dir> --out_dir <out_dir>

# Extract mel/lin spectrogram and dictionary of characters/phonemes
python3 preprocess.py --txt_dir <txt_dir> --wav_dir <wav_dir> --bin_dir <bin_dir>

# train the model, --use_txt will control vc path or tts path
python3 main.py -m <message> -g <gpu_id> --use_txt <0~1, higher value means y_t batch is more sampled>
```

