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
import fourier_messager as fm

(audio, sr) = fm.load_sound("sample/audio.wav") 
# specify device if cuda not installed or not supported
#(audio, sr) = fm.load_sound("sample/audio.wav", device='cpu') 
```

Next, generate a spectrogram image of the audio to create a target spectrogram.
Alternate functions can be used, so long as they are tensor gradient friendly
for training.

```python
target_spec = fm.mel_spectrogram(audio, sr=sr)
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
target_spec = fm.add_img_to_spec(target_spec, "sample/hello-world.png", 35, 50)
```

Once the target spectrogram has been created, training can start.

```python
config = fm.AudioTrainerConfig(sr=sr)
trainer = fm.HiddenStereoAudioTrainer(audio, target_spec, config)

new_audio = trainer.train()

# save modified waveform
import torchaudio

torchaudio.save("audio-modified.wav", new_audio, sr)
```

`HiddenStereoAudioTrainer` will take a mono track and generate a stereo track.
a spectrogram of both channels in the stereo track should be similar to the
spectrogram of the base audio, but a spectrogram of the channels added together
will be close to the target spectrogram.

a modified spectrogram image can be saved using `save_spectrogram`

```python
from fourier_messager.visualization import save_spectrogram

save_spectrogram("spectrogram.png", fm.mel_spectrogram(new_audio))
```