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
    audionp = trim_silence(np.float32(audionp))
    audio = torch.tensor(audionp, device=device)
    if (normalize) :
        audio = audio / torch.max(audio)
    return audio, sr

def trim_silence(audio : torch.Tensor) :
    start = 0
    end = len(audio) - 1
    while audio[start].item() == 0 :
        start += 1
    while audio[end].item() == 0 :
        end -= 1
    return audio[start:(end + 1)]

def get_audio_model(audio : torch.Tensor) :
    ret = torch.nn.Linear(1, audio.shape[0], bias=False, device=audio.device)
    
    ret.weight.data[:, 0] = audio
    return ret