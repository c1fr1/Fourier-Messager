from torchaudio.transforms import MelSpectrogram
import torch

from fourier_messager.util import load_greyscale_img

def mel_spectrogram(audio : torch.Tensor, sr=44100, f_min=0, f_max=8000, n_fft=2048, n_mels=128, hop_length=512) :
    transform = MelSpectrogram(
        sample_rate=sr, 
        f_min=f_min, 
        f_max=f_max, 
        normalized=True, 
        n_fft=n_fft, 
        n_mels=n_mels, 
        hop_length=hop_length,
    ).to(device=audio.device)

    spec = transform(audio)
    if len(spec.shape) == 2 :
        spec = spec[:, 2:(spec.shape[1] - 2)]
    elif len(spec.shape) == 3 :
        spec = spec[:, :, 2:(spec.shape[2] - 2)]
    spec = 10 * torch.log10(spec)
    return spec

def add_img_to_spec(
        base_spec : torch.Tensor,
        image : str | torch.Tensor,
        startx : int,
        starty : int,
        type : str = 'subtract',
):
    if isinstance(image, str) :
        image = load_greyscale_img(image, device=base_spec.device)
    image = image.flip(0)

    spec_min = torch.min(base_spec)
    maxx = startx + image.shape[1]
    maxy = starty + image.shape[0]

    if (maxx > base_spec.shape[1] or startx < 0) :
        raise RuntimeError(f"x range ({startx}:{maxx}) out of spectrogram range (0:{base_spec.shape[1]})")
    if (maxy > base_spec.shape[0] or starty < 0) :
        raise RuntimeError(f"x range ({starty}:{maxy}) out of spectrogram range (0:{base_spec.shape[0]})")
    
    if type == 'subtract' :
        base_spec[starty:maxy, startx:maxx] = base_spec[starty:maxy, startx:maxx] + (spec_min - base_spec[starty:maxy, startx:maxx]) * image
    elif type == 'paste' :
        spec_range = torch.max(base_spec) - torch.min(base_spec)
        base_spec[starty:maxy, startx:maxx] *= 0
        base_spec[starty:maxy, startx:maxx] += spec_min
        base_spec[starty:maxy, startx:maxx] += spec_range * image
    return base_spec
