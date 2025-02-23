from typing import Callable

from torch.utils.data import Sampler
from transformers import Trainer

RewardFunc = Callable[[list, list], list[float]]

class VLMR1Trainer(Trainer):
    '''
    Custom GRPO trainer for VLM R1. Based off of TRLs GRPOTrainer and my TRL fork, but specifically focused on
    what is needed for this project. It /will/ not support all possibile training configurations,
    but instead will be hyperforcused on the needs of this project.
    '''
    
    def __init__(self):
        raise NotImplementedError()
    
    
    def _set_signature_columns_if_needed(self):
        raise NotImplementedError()
    
    def _get_train_sampler(self) -> Sampler:
        raise NotImplementedError()
    
    def _move_model_to_vllm(self):
        raise NotImplementedError()
    
    
    def _get_log_probs(self):
        raise NotImplementedError()
    
    def _prepare_inputs(self):
        raise NotImplementedError()

    
    def compute_loss(self):
        raise NotImplementedError()
    
    def prediction_step(self):
        raise NotImplementedError()
    
    def log(self):
        raise NotImplementedError()

    
    
    