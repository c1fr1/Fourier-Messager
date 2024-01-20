# Fourier Messager

A tool for embedding messages/images into spectrograms of waveforms while
preserving quality of the waveform.

# Dependencies

All Dependencies are listed in [requirements.txt](requirements.txt). Versions 
very likely have more flexibility than specified, but it is reccomended to use 
cuda for PyTorch, which may require manual installation. This project was
tested on Python 3.12.1, which does require the nightly build of PyTorch (2.3)
as of 20/1/2023.

```bash
pip install -r requirements.txt
```

The project can still be run with no CUDA support, but `device='cpu'` will need
to be specified when loading audio.

# Usage

Load the audio file containing the base waveform.

44.1 khz audio was used for testing and serves as the default in a few
functions, but other sample rates should work.

```python
from fourier_messager import load_sound, mel_spectrogram, add_img_to_spec

(audio, sr) = load_sound("audio.wav") 
# specify device if cuda not installed or not supported
#(audio, sr) = load_sound("audio.wav", device='cpu') 
```

Next, generate a spectrogram image of the audio to create a target spectrogram.
Alternate functions can be used, so long as they are tensor gradient friendly
for training.

```python
target_spec = mel_spectrogram(audio, sr=sr)
```

The spectrogram can be modified by any means, but functions `add_image_to_spec`
is supplied to put images on the spectrogram. Reducing values in the
spectrogram has a less noticeable affect on the output audio than increasing
them.

Either a path to an image file can specified, or a 2d tensor. If a tensor is
supplied, larger values will be subtracted from the spectrogram, and lower
values (closer to 0) will leave the spectrogram unchanged. All modifications
stay in the range of values present in the input spectrogram. If a path is
specified, the image will be converted to a greyscale image where black is 1.0,
and white is 0.0 before doing the process above.

```python
target_spec = add_img_to_spec(target_spec, "image.png", 35, 50)
#final path
```

Once the target spectrogram has been created, training can start.

```python
new_audio = train_model(audio, sr, target_spec)

# save modified waveform
import torch, torchaudio

new_audio = torch.unsqueeze(new_audio.detach(), 0).cpu()
torchaudio.save("audio-modified.wav", new_audio, sr)
```

## Key Optional training parameters:

### epochs

More epochs = more good, but takes longer.

### wav_spec_loss_ratio

Ratio of how heavily weighted preserving the audio is to achieving the target
spectrogram. Values closer to 1 will maintain the input waveform, while values
closer to 0 will create an image closer to the target spectrogram.

### step_fn : Callable[[float, float, float], any]

Function that can be used to record loss values, (wav_loss, spec_loss, 
aggregate_loss).

a modified spectrogram image can be saved using `save_spectrogram`

```python
from fourier_messager.visualization import save_spectrogram

save_spectrogram("spectrogram.png", mel_spectrogram(new_audio))
```

# Training notes

The loss function used is effectiely a weighted sum of `wav_loss` and
`spec_loss`. `wav_loss` represents the similarity of the candidate solution to
the input waveform and is proportional to the mean squared error of the two.
Similarly, `spec_loss` is proportional to the mean squared error of the
spectrogram of the candidate solution and the target spectrogram. `spec_loss`
has a much noiser/bumpier surface, but the `wav_loss` is much smoother. This
results in higher learning rates doing well at creating the defining features
of the spectrogram, while lower learning rates do a better job of balancing the
two goals, but can fail to catch larger features in the spectrogram. To utilise
this tradeoff, a decaying learning rate was used on top of Adam (the default
primary optimisation function). In the main training function, a starting 
learning rate can be specified along with a minimum learning rate. An
exponential decay rate will be calculated to interpolate between those learning
rates. If the minimum learning rate is set to None, the learning rate will not
change with training.

An attempt has been made to normalise both `spec_loss` and `wav_loss` so that
they have similar values no matter what the input waveform/spectrogram is, but
some parameter tuning may still achieve better results depending on the input.