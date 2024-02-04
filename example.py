import torch
import torchaudio

import matplotlib.pyplot as plt

import numpy as np

import fourier_messager as fm
from fourier_messager.visualization import save_spectrogram

if __name__ == "__main__" :
    (audio, sr) = fm.load_sound("sample/audio.wav")
    
    target_spec = fm.mel_spectrogram(audio, sr=sr)
    save_spectrogram("figures/base-spectrogram.png", target_spec)
    target_spec = fm.add_img_to_spec(target_spec, "sample/hello-world.png", 110, 105)
    save_spectrogram("figures/target-spectrogram.png", target_spec)

    wav_losses_a = []
    wav_losses_b = []
    spec_losses_a = []
    spec_losses_b = []
    mean_spec_losses = []
    ag_losses = []

    def step_fn(
            wav_loss_a : float, 
            wav_loss_b : float, 
            spec_loss_a : float, 
            spec_loss_b : float, 
            mean_spec_loss : float,
            ag_loss : float) :
        
        wav_losses_a.append(wav_loss_a.item())
        wav_losses_b.append(wav_loss_b.item())
        spec_losses_a.append(spec_loss_a.item())
        spec_losses_b.append(spec_loss_b.item())
        mean_spec_losses.append(mean_spec_loss.item())
        ag_losses.append(ag_loss.item())

    trainer = fm.HiddenStereoAudioTrainer(
        audio, 
        target_spec, 
        fm.AudioTrainerConfig(sr=sr, lr=0.0015, epochs=1e5, lr_min=1.0e-6),
        loss_weights=torch.tensor([2.0, 2.0, 1.0, 1.0, 6.0]))
    trainer.add_step_callback(lambda losses, ag_loss : step_fn(*losses, ag_loss))

    new_audio = trainer.train()

    wav_losses_a = np.array(wav_losses_a)
    wav_losses_b = np.array(wav_losses_b)
    spec_losses_a = np.array(spec_losses_a)
    spec_losses_b = np.array(spec_losses_b)
    mean_spec_losses = np.array(mean_spec_losses)
    ag_losses = np.array(ag_losses)

    print(f"ag_loss     : {ag_losses[-1]}")
    print(f"mean_loss   : {mean_spec_losses[-1]}")
    print(f"spec_a_loss : {spec_losses_a[-1]}")
    print(f"spec_b_loss : {spec_losses_b[-1]}")
    print(f"wav_a_loss  : {wav_losses_a[-1]}")
    print(f"wav_b_loss  : {wav_losses_b[-1]}")
    
    #save spectrograms
    spec_a = fm.mel_spectrogram(new_audio[0])
    spec_b = fm.mel_spectrogram(new_audio[1])
    mean_audio = new_audio.mean(0)
    mean_spec = fm.mel_spectrogram(mean_audio)
    save_spectrogram("figures/spec_a.png", spec_a)
    save_spectrogram("figures/spec_b.png", spec_b)
    save_spectrogram("figures/mean_spec.png", mean_spec)
    mean_audio = torch.unsqueeze(mean_audio.detach(), 0).cpu()
    torchaudio.save("sample/encoded-stereo.wav", new_audio.detach().cpu(), sr)
    torchaudio.save("test-audio/stereo-channel-average.wav", mean_audio, sr)
    
    #plot losses
    fig, axd = plt.subplot_mosaic([['ag', 'ag'],
                                   ['ag', 'ag'],
                                   ['mean', 'mean'],
                                   ['spec_a', 'spec_b'],
                                   ['wav_a', 'wav_b']], layout='constrained')

    fig.set_size_inches(9.6, 6.4)

    x = np.linspace(1, len(ag_losses), len(ag_losses))

    for ax_name in axd :
        axd[ax_name].set_yscale('log')

    axd['ag'].set_title('Aggregate Loss')
    axd['mean'].set_title('Mean Channel Spectrogram Loss')
    axd['spec_a'].set_title('Channel A Spectrogram Loss')
    axd['spec_b'].set_title('Channel B Spectrogram Loss')
    axd['wav_a'].set_title('Channel A Waveform Loss')
    axd['wav_b'].set_title('Channel B Waveform Loss')

    axd['ag'].plot(x, ag_losses)
    axd['mean'].plot(x, mean_spec_losses)
    axd['spec_a'].plot(x, spec_losses_a)
    axd['spec_b'].plot(x, spec_losses_b)
    axd['wav_a'].plot(x, wav_losses_a)
    axd['wav_b'].plot(x, wav_losses_b)
    
    fig.savefig("figures/losses.png")