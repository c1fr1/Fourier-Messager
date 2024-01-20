import torch
import torchaudio

import matplotlib.pyplot as plt

import numpy as np

from util import load_sound
from spectrograms import mel_spectrogram, add_img_to_spec, train_model
from training import train_model
from visualization import save_spectrogram

if __name__ == "__main__" :
    (audio, sr) = load_sound("test-audio/invoker99.mp3")
    
    target_spec = mel_spectrogram(audio, sr=sr)
    target_spec = add_img_to_spec(target_spec, "test-audio/any.png", 35, 50)
    target_spec = add_img_to_spec(target_spec, "test-audio/eyes.png", 135, 50)
    save_spectrogram("test-audio/target-spec.wav.png", target_spec)

    wav_losses = []
    spec_losses = []
    total_losses = []

    def step_fn(wav_loss, spec_loss, total_loss) :
        wav_losses.append(wav_loss)
        spec_losses.append(spec_loss)
        total_losses.append(total_loss)

    new_audio = train_model(
        audio,
        sr, 
        target_spec,
        lr=0.01,
        lr_min=0.000000015,
        epochs=10000,
        wav_spec_loss_ratio=0.5,
        step_fn=step_fn)
    
    new_spec = mel_spectrogram(new_audio)
    new_audio = torch.unsqueeze(new_audio.detach(), 0).cpu()
    torchaudio.save("test-audio/invoker99-new.wav", new_audio, sr)
    save_spectrogram("test-audio/invoker99-new.png", new_spec)
    
    #plot losses
    wav_losses = np.array(wav_losses)
    spec_losses = np.array(spec_losses)
    total_losses = np.array(total_losses)

    fig, axd = plt.subplot_mosaic([['ag', 'ag'],
                                   ['wav', 'spec']], layout='constrained')

    fig.set_size_inches(9.6, 6.4)

    x = np.linspace(1, len(total_losses), len(total_losses))

    axd['ag'].set_title('Aggregate Loss')
    axd['wav'].set_title('Wav Loss')
    axd['spec'].set_title('Spectrogram Loss')

    axd['ag'].plot(x, total_losses)
    axd['wav'].plot(x, wav_losses)
    axd['spec'].plot(x, spec_losses)
    
    fig.savefig("test-audio/losses.png")

    print(f"ag_loss   : {total_losses[-1]}")
    print(f"wav_loss  : {wav_losses[-1]}")
    print(f"spec_loss : {spec_losses[-1]}")