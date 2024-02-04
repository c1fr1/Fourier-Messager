import torch
from fourier_messager.trainers import AudioTrainer, AudioTrainerConfig
from fourier_messager.util import get_audio_model
from fourier_messager.loss_functions import WaveformLoss, SpectrogramLoss

class MonoAudioTrainer(AudioTrainer) :
    def __init__(
            self, 
            base_audio : torch.Tensor,
            target_spec : torch.Tensor,
            config : AudioTrainerConfig = AudioTrainerConfig,
            wav_spec_ratio : torch.Tensor | None = None, 
            optimizer : torch.nn.Module | None = None) -> None:
        
        self.wav_spec_loss_ratio = wav_spec_ratio
        loss_weights = torch.Tensor([wav_spec_ratio, 1 - wav_spec_ratio])
        
        model = get_audio_model(base_audio)
        super().__init__(
            model, 
            [WaveformLoss(base_audio), SpectrogramLoss(target_spec)], 
            config, 
            loss_weights, 
            optimizer)

