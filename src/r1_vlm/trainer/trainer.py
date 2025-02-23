from typing import Callable, List, Optional, Union

import torch
from torch.utils.data import Sampler
from transformers import (
    PreTrainedModel,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Trainer,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from trl import GRPOConfig

RewardFunc = Callable[[list, list], list[float]]


class VLMR1Trainer(Trainer):
    """
    Custom GRPO trainer for VLM R1. Based off of TRLs GRPOTrainer and my TRL fork, but specifically focused on
    what is needed for this project. It /will/ not support all possibile training configurations,
    but instead will be hyperforcused on the needs of this project.

    Args:
    model: PreTrainedModel - The model to train.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        args: GRPOConfig,
        reward_funcs: Union[RewardFunc, List[RewardFunc]],
    ):
        model_init_kwargs = args.model_init_kwargs or {}
        model_id = model.config._name_or_path

        # init ref model is using zero3
        if is_deepspeed_zero3_enabled():
            model_init_kwargs_dict = model_init_kwargs.__dict__

            ref_model_path = model_init_kwargs_dict["model_name_or_path"]
            ref_model_torch_dtype = model_init_kwargs_dict["torch_dtype"]

            attn_implementation = model_init_kwargs_dict["attn_implementation"]

            if "Qwen2-VL" in ref_model_path:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    ref_model_path,
                    torch_dtype=ref_model_torch_dtype,
                    attn_implementation=attn_implementation,
                )
            elif "Qwen2.5-VL" in ref_model_path:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    ref_model_path,
                    torch_dtype=ref_model_torch_dtype,
                    attn_implementation=attn_implementation,
                )
            else:
                raise ValueError(
                    "The base model you provided was unexpected. Expected a Qwen2-VL or Qwen2.5-VL."
                )
            self.ref_model.use_cache = False

        else:
            self.ref_model = None

        # handle reward funcs 
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_funcs = reward_funcs
        
        
        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(len(args.reward_weights))}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)
        
        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features
        
        # Training arguments
        if args.max_prompt_length is not None:
            raise ValueError("max_prompt_length is not supported")
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        
        self.beta = args.beta
        # made it to line 350 of my forked trainer...

        raise NotImplementedError("Haven't finished init")

    def _set_signature_columns_if_needed(self):
        raise NotImplementedError()

    def _get_train_sampler(self) -> Sampler:
        raise NotImplementedError()

    def _move_model_to_vllm(self):
        raise NotImplementedError()

    def _get_log_probs(self):
        raise NotImplementedError()

    def _prepare_inputs(self):
        """
        1. Use Environment to generate EnvironmentOutput
        2. Compute log probs
        3. Compute rewards
        4. Compute advantages
        """

        # TODO:
        # []  Use Environment to generate EnvironmentOutput
        # []  Compute log probs
        # []  Compute rewards
        # []  Compute advantages
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
        raise NotImplementedError("Need to implement compute_loss")
        # TODO: uncomment this when ready!
        # with torch.no_grad():
        #    with self.compute_loss_context_manager():
        #        loss = self.compute_loss(model, inputs)
        #    loss = loss.mean().detach()
        # return loss, None, None

    def log(self):
        raise NotImplementedError()
