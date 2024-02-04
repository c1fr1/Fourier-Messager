import torch
from torch.optim import Adam

import numpy as np

from tqdm import tqdm

class AudioTrainerConfig :
    def __init__(
            self,
            sr : int = 44100,
            lr : float = 0.01,
            lr_min : float = 1.5e-8,
            epochs : int = 10000) -> None:
        self.sr = sr
        self.lr = lr
        self.lr_min = lr_min
        self.epochs = int(epochs)
        pass

class AudioTrainer :
    def __init__(
            self, 
            model : torch.nn.Module, 
            loss_fns : list,
            config : AudioTrainerConfig = AudioTrainerConfig,
            loss_weights : torch.Tensor | None = None,
            optimizer : torch.nn.Module | None = None) -> None :

        self.model = model
        self.config = config
        self.loss_fns = loss_fns
        if (loss_weights == None) :
            self.loss_weights = torch.tensor([1.0 / len(loss_fns)] * len(loss_fns))
        else :
            self.loss_weights = loss_weights / loss_weights.sum()
        if (optimizer == None) :
            self.optimizer = Adam(model.parameters(), config.lr)
        else :
            self.optimizer = optimizer
        self.step_callbacks = []

    def train(self, use_tqdm : bool = True) :

        lr_decay = np.exp(np.log(self.config.lr_min / self.config.lr) / self.config.epochs)
        
        iter_range = range(self.config.epochs) 
        if (use_tqdm) :
            iter_range = tqdm(iter_range)

        for _ in iter_range :
            self.step()
            self.optimizer.param_groups[0]['lr'] *= lr_decay
        return self.prep_losses(self.model)

    def prep_losses(self, candidate_model : torch.nn.Module) -> torch.Tensor :
        return candidate_model(torch.tensor([1.0], device=next(candidate_model.parameters()).device))

    def step(self) :
        current_audio = self.prep_losses(self.model)
        losses, total_loss = self.calculate_losses(current_audio)

        self.on_step(losses, total_loss)
        for f in self.step_callbacks :
            f(losses, total_loss)
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def on_step(self, losses : torch.Tensor, ag_loss : torch.Tensor) :
        pass
    
    def calculate_losses(self, current_audio : torch.Tensor) :
        losses = torch.empty(len(self.loss_fns))
        for i in range(len(self.loss_fns)) :
            losses[i] = self.loss_fns[i](current_audio)
        total_loss = losses.dot(self.loss_weights)
        return losses, total_loss

    def add_step_callback(self, f) :
        self.step_callbacks.append(f)
        