from typing import Callable

from torch.utils.data import Sampler
from transformers import Trainer

from r1_vlm.trainer.interfaces import EnvironmentInput, EnvironmentOutput

EnvironmentInput
EnvironmentOutput

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
    
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys: Optional[list[str]] = None,
    ):
        inputs = self._prepare_inputs(inputs)
        #with torch.no_grad():
        #    with self.compute_loss_context_manager():
        #        loss = self.compute_loss(model, inputs)
        #    loss = loss.mean().detach()
        #return loss, None, None
    
    def log(self):
        raise NotImplementedError()

    
    
        