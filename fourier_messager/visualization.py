import torch
import matplotlib.pyplot as plt

def save_spectrogram(path : str, spec : torch.Tensor, name="Spectrogram", hop_length=512, sr=44100) :
    fig, ax = plt.subplots()
    ax.set_title(name)
    ax.imshow(spec.detach().cpu().numpy(), origin='lower', aspect='auto', extent=[0, spec.shape[1] * hop_length / sr, 0, spec.shape[0]])
    fig.savefig(path)