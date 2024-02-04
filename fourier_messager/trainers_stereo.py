import torch

from fourier_messager.spectrograms import mel_spectrogram
from fourier_messager.trainers import AudioTrainerConfig, AudioTrainer
from fourier_messager.util import get_audio_model_stereo
from fourier_messager.loss_functions import WaveformLoss, SpectrogramLoss, CombinedChannelLoss

from torch.optim import Adam

class StereoAudioTrainer(AudioTrainer) :
    def __init__(
            self, 
            model : torch.nn.Module, 
            channel_loss_fns : list,
            audio_loss_fns : list,
            config : AudioTrainerConfig = AudioTrainerConfig,
            channels : int = 2,
            loss_weights : torch.Tensor | None = None,
            optimizer : torch.nn.Module | None = None) -> None:
        
        self.model = model
        self.config = config
        self.channel_loss_fns = channel_loss_fns
        self.audio_loss_fns = audio_loss_fns
        self.loss_fns = channel_loss_fns + audio_loss_fns
        self.channels = channels
        if (loss_weights == None) :
            loss_fn_count = channels * len(channel_loss_fns) + len(audio_loss_fns)
            self.loss_weights = torch.tensor([1.0 / loss_fn_count] * loss_fn_count)
        else :
            self.loss_weights = loss_weights / loss_weights.sum()
        if (optimizer == None) :
            self.optimizer = Adam(model.parameters(), config.lr)
        else :
            self.optimizer = optimizer
        self.step_callbacks = []

    def prep_losses(self, candidate_model : torch.nn.Module) -> torch.Tensor :
        output = candidate_model(torch.tensor([1.0], device=next(candidate_model.parameters()).device))
        return output.unflatten(0, (2, int(output.shape[0] / 2)))
    
    def calculate_losses(self, current_audio : torch.Tensor) :
        losses = torch.empty(len(self.channel_loss_fns) * self.channels + len(self.audio_loss_fns))
        for i in range(len(self.channel_loss_fns)) :
            loss_vals = self.channel_loss_fns[i](current_audio)
            for j in range(self.channels) :
                losses[i * self.channels + j] = loss_vals[j]
        for i in range(len(self.audio_loss_fns)) :
            losses[i + self.channels * len(self.channel_loss_fns)] = self.audio_loss_fns[i](current_audio)
        return losses, losses.dot(self.loss_weights)
    
class HiddenStereoAudioTrainer(StereoAudioTrainer) :
    def __init__(
            self, 
            audio : torch.Tensor, 
            target_spec : torch.Tensor,
            config : AudioTrainerConfig = AudioTrainerConfig,
            loss_weights : torch.Tensor | None = None,
            optimizer : torch.nn.Module | None = None) -> None:
        
        if (len(audio.shape) == 2) :
            audio = audio.unsqueeze(0)
        if (audio.shape[0] == 1) :
            audio = audio.repeat(2, 1)
        base_spec = mel_spectrogram(audio, sr=config.sr)
        model = get_audio_model_stereo(audio)
        super().__init__(
            model,
            [WaveformLoss(audio), SpectrogramLoss(base_spec, sr=config.sr)],
            [CombinedChannelLoss(SpectrogramLoss(target_spec, sr=config.sr))],
            config,
            2,
            loss_weights,
            optimizer)
