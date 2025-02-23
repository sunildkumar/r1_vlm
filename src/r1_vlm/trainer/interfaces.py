from typing import Callable

from pydantic import BaseModel
from torch import Tensor


class EnvironmentInput(BaseModel):
    '''
    Interface for what an environment's .generate() should accept.
    '''
    
    # list of examples from the dataset
    examples: list[dict]
    

class EnvironmentOutput(BaseModel):
    '''
    Interface for what an environment's .generate() should return. 
    '''
    
    # tensor of shape (num_completions, len of completion)
    completion_ids: Tensor
        
    # tensor of shape (num_completions, len of prompt + len of completion)
    prompt_completion_ids: Tensor
        
    # tensor of shape (num_completions, len of completion)
    completion_loss_mask: Tensor
        
    # tensor of shape (num_completions, len of prompt + len of completion)
    attention_mask: Tensor
        
    # image data prepared by the processor, length of list is num_completions
    multi_modal_data: list[dict]
        
    # list of length num_completions
    completions_text: list[str]


class LogProbsInput(BaseModel):
    '''
    Interface for what log probs computation should accept.
    '''
    
    # tensor of shape (num_completions, len of completion)
    completion_ids: Tensor
    
    # prompt completion ids, tensor of shape (num_completions, len of prompt + len of completion)
    prompt_completion_ids: Tensor
    
    # attention mask, tensor of shape (num_completions, len of prompt + len of completion)
    attention_mask: Tensor
    
    # image data prepared by the processor, length of list is num_completions
    multi_modal_data: list[dict]
    
class LogProbsOutput(BaseModel):
    '''
    Interface for what log probs computation should return. 
    '''
    # tensor of shape (num_completions, len of completion)
    per_token_log_probs: Tensor
    
    # tensor of shape (num_completions, len of completion)
    ref_per_token_log_probs: Tensor
    

class AdvantagesInput(BaseModel):
    '''
    Interface for what advantages computation should accept.
    '''
    # list of examples from the dataset
    examples: list[dict]
    
    # list of len num_completions
    completions_text: list[str]
    
    # list of reward functions
    reward_fns: list[Callable]

class AdvantagesOutput(BaseModel):
    '''
    Interface for what advantages computation should return.
    '''
    
    # tensor of shape (num_completions, )
    advantages: Tensor
    

class ComputeLossInput(BaseModel):
    '''
    Interface for what compute loss computation should accept.
    '''
    
    # tensor of shape (num_completions, len of completion)
    per_token_log_probs: Tensor
    
    # tensor of shape (num_completions, len of completion)
    ref_per_token_log_probs: Tensor
    
    # tensor of shape (num_completions, )
    advantages: Tensor
    
    # tensor of shape (num_completions, len of completion)
    completion_loss_mask: Tensor

class ComputeLossOutput(BaseModel):
    '''
    Interface for what compute loss computation should return.
    '''
    # scalar loss
    loss: float
    
    
  
    