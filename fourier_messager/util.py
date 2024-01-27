from matplotlib.pyplot import imread
import numpy as np
import torch
import soundfile

def load_greyscale_img(path : str, device : str = 'cuda') :
    image = imread(path)
    if len(image.shape) > 2 :
        if (image.shape[2] == 4) :
            image[:, :, 0:3] *= image[:, :, 3:4]
            image = image[:, : , 0:3]
        image = np.mean(image, 2)
    image = 1 - image
    return torch.tensor(image, device=device)

def load_sound(path, normalize : bool = True, device : str = 'cuda') :
    (audionp, sr) = soundfile.read(path)
    if len(audionp.shape) == 2 :
        audionp = audionp.transpose((1, 0))
    audionp = np.float32(audionp)
    audio = torch.tensor(audionp, device=device)
    audio = trim_silence(audio)
    if (normalize) :
        audio = audio / torch.max(audio)
    return audio, sr

def trim_silence(audio : torch.Tensor) :
    start = 0
    if len(audio.shape) == 2 :
        return trim_silence_stereo(audio)
    end = audio.shape[0] - 1
    while audio[start].item() == 0 :
        start += 1
    while audio[end].item() == 0 :
        end -= 1
    return audio[start:(end + 1)]

def trim_silence_stereo(audio : torch.Tensor) :
    start = 0
    end = audio.shape[1] - 1
    while audio[0][start] == 0 and audio[1][start] == 0 :
        start += 1
    while audio[0][end] == 0 and audio[0][end] == 0 :
        end -= 1
    return audio[:, start:(end + 1)]

def get_audio_model(audio : torch.Tensor) :
    ret = torch.nn.Linear(1, audio.shape[0], bias=False, device=audio.device)
    
    ret.weight.data[:, 0] = audio
    return ret

def get_audio_model_stereo(audio : torch.Tensor, other_channel : torch.Tensor | None = None) :
    ret = torch.nn.Linear(1, audio.shape[0] * 2, bias=False, device=audio.device)
    if (other_channel == None) :
        other_channel = audio.clone()
    ret.weight.data[0:audio.shape[0], 0] = audio
    ret.weight.data[audio.shape[0]:(audio.shape[0] * 2), 0] = other_channel
    return ret