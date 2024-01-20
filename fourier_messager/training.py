import torch
from torch.nn.functional import mse_loss as mse
from torch.optim import Adam

import numpy as np

from tqdm import tqdm

from spectrograms import mel_spectrogram
from util import get_audio_model

from typing import Callable

def loss_fns(
        base_audio : torch.Tensor,
        estimate : torch.Tensor,
        sr : int,
        target_spec : torch.Tensor,
        max_spec_loss : float,
        max_wav_loss : float,
        wav_spec_loss_ratio : float = 0.5,
        spectrogram_fn : Callable[[torch.Tensor, int], torch.Tensor] = lambda audio, sr : mel_spectrogram(audio, sr=sr)) :
    
    current_spec = spectrogram_fn(estimate, sr)

    wav_loss = mse(estimate, base_audio) / max_wav_loss
    spec_loss = mse(current_spec, target_spec) / max_spec_loss
    aggregate_loss = wav_loss * wav_spec_loss_ratio + spec_loss * (1 - wav_spec_loss_ratio)

    return wav_loss, spec_loss, aggregate_loss

def train_model_step(
        model : torch.nn.Module, 
        base_audio : torch.Tensor,
        target_spec : torch.Tensor,
        optimizer : torch.optim.Optimizer,
        max_spec_loss : float,
        max_wav_loss : float,
        sr : int,
        wav_spec_loss_ratio : float = 0.5,
        spectrogram_fn : Callable[[torch.Tensor, int], torch.Tensor] = lambda audio, sr : mel_spectrogram(audio, sr=sr)) :
    
    current_audio = model(torch.tensor([1.0], device=base_audio.device))

    wav_loss, spec_loss, ag_loss = loss_fns(
        base_audio, 
        current_audio,
        sr,
        target_spec,
        max_spec_loss,
        max_wav_loss,
        wav_spec_loss_ratio,
        spectrogram_fn=spectrogram_fn)

    optimizer.zero_grad()
    ag_loss.backward()
    optimizer.step()

    return wav_loss.item(), spec_loss.item() , ag_loss.item()

def train_model(
        base_audio : torch.Tensor,
        sr : int,
        target_spec : torch.Tensor,
        lr : float = 0.01,
        epochs : int = 10000,
        wav_spec_loss_ratio = 0.5,
        lr_min : float | None = 1.5e-08,
        optimizer : torch.optim.Optimizer = None,
        use_tqdm : bool = True,
        step_fn : Callable[[float, float, float], any] | None = None,
        spectrogram_fn : Callable[[torch.Tensor, int], torch.Tensor] = lambda audio, sr : mel_spectrogram(audio, sr=sr)) :
    
    model = get_audio_model(base_audio)

    if optimizer == None :
        optimizer = Adam(model.parameters(), lr)

    if lr_min == None :
        lr_decay = 1
    else :
        lr_decay = np.exp(np.log(lr_min / lr) / epochs)
        print(f"lr decay rate: {lr_decay}")

    base_spec = spectrogram_fn(base_audio, sr)

    max_spec_loss = mse(base_spec, target_spec)
    max_wav_loss = torch.var(base_audio) * max_spec_loss / torch.var(base_spec)

    iter_range = range(epochs) 
    if (use_tqdm) :
        iter_range = tqdm(iter_range)

    for _ in iter_range :
        losses = train_model_step(
            model, 
            base_audio, 
            target_spec, 
            optimizer, 
            max_spec_loss, 
            max_wav_loss, 
            sr, 
            wav_spec_loss_ratio,
            spectrogram_fn=spectrogram_fn)
        if (step_fn != None) :
            step_fn(*losses)
        optimizer.param_groups[0]['lr'] *= lr_decay

    new_audio : torch.Tensor = model(torch.tensor([1.0], device=base_audio.device))

    return new_audio