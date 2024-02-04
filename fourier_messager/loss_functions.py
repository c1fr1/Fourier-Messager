from fourier_messager.spectrograms import mel_spectrogram
import torch

class WaveformLoss(torch.nn.Module) :
    def __init__(self, base_audio : torch.Tensor) -> None:
        super(WaveformLoss, self).__init__()
        self.base_audio = base_audio
    def forward(self, x : torch.Tensor) :
        dims = max(len(x.shape), len(self.base_audio.shape))
        return (self.base_audio - x).pow(2).mean(dims - 1)
    
class SpectrogramLoss(torch.nn.Module) :
    def __init__(
            self, 
            target_spec : torch.Tensor,
            sr = 44100, 
            f_min = 0, 
            f_max = 8000, 
            n_fft = 2048, 
            n_mels = 128, 
            hop_length = 512) -> None:
        super(SpectrogramLoss, self).__init__()
        self.target_spec = target_spec
        self.sr = sr
        self.f_min = f_min
        self.f_max = f_max
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length

    def forward(self, x : torch.Tensor) :
        spec = mel_spectrogram(
            x,
            sr=self.sr,
            f_min=self.f_min,
            f_max=self.f_max,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            hop_length=self.hop_length)
        dims = max(len(spec.shape), len(self.target_spec.shape))
        return (self.target_spec - spec).pow(2).mean((dims - 1, dims - 2))
    
class CombinedChannelLoss(torch.nn.Module) :
    def __init__(self, internal_f) -> None:
        super().__init__()
        self.internal_f = internal_f
    
    def forward(self, x : torch.Tensor) :
        return self.internal_f(x.mean(0))