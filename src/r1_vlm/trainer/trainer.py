import warnings
from collections import defaultdict
from typing import Callable, List, Optional, Union
from unittest.mock import patch

import torch
from accelerate.utils import set_seed
from datasets import Dataset, IterableDataset
from torch.utils.data import Sampler
from transformers import (
    AutoProcessor,
    PreTrainedModel,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Trainer,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from trl import GRPOConfig
from vllm import LLM, SamplingParams

RewardFunc = Callable[[list, list], list[float]]


class VLMR1Trainer(Trainer):
    """
    Custom GRPO trainer for VLM R1. Based off of TRLs GRPOTrainer and my TRL fork, but specifically focused on
    what is needed for this project. It /will/ not support all possibile training configurations,
    but instead will be hyperforcused on the needs of this project.

    Args:
    model: PreTrainedModel - The model to train.
    args: GRPOConfig -  Training arguments.
    reward_funcs: Union[RewardFunc, List[RewardFunc]] - The reward functions to use.
    train_dataset: Optional[Union[Dataset, IterableDataset]] - The training dataset.
    processing_class: AutoProcessor - The processor to use.
    # TODO: Define a type for the environment
    env  - The environment to use.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        args: GRPOConfig,
        reward_funcs: Union[RewardFunc, List[RewardFunc]],
        train_dataset: Union[Dataset, IterableDataset],
        processing_class: AutoProcessor,
        env,

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
        
        
        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True
        
        # Initialize the metrics
        self._metrics = defaultdict(list)
        self.log_completions = args.log_completions
        
        
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            processing_class=processing_class,
        )
        
        
        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed. We can probably skip this as we use vLLM, but
        # it's safer to set just in case.
        set_seed(args.seed, device_specific=True)
        
        
        if self.accelerator.is_main_process:
            vllm_device = self.args.vllm_device
            if vllm_device == "auto":
                if torch.cuda.device_count() == 1:
                    print("Only one GPU available, sharing it between vLLM and training.")
                    vllm_device = "cuda:0"  # particular case when training with onyl 1 GPU: share it
                else:
                    vllm_device = f"cuda:{self.accelerator.num_processes}"  # take the next GPU idx
                    print(f"Using GPU {vllm_device} for vLLM.")

            # Check that the requested device is available
            if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
                raise ValueError(
                    f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                    "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                    "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                    f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                )
            # Check that the requested device is not also used for training
            if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
                warnings.warn(
                    f"The requested device {vllm_device} is also being used for training. For higher throughput "
                    "and to avoid out-of-memory errors, it is recommended to use a dedicated device for vLLM. "
                    "If this is intentional, you may ignore this warning but should adjust "
                    "`vllm_gpu_memory_utilization` accordingly."
                )
            # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
            # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
            # setting (profiling_patch).
            world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
            profiling_patch = patch(
                "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
                return_value=None,
            )
            with world_size_patch, profiling_patch:
                self.vlm = LLM(
                    model=model.name_or_path,
                    device=vllm_device,
                    gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                    dtype=self.args.vllm_dtype,
                    # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                    # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                    # This is particularly useful here because we generate completions from the same prompts.
                    enable_prefix_caching=True,
                    max_model_len=self.args.vllm_max_model_len,
                    # Setting this to 1 as we only have one image per prompt for now. Setting it longer requires more resources, which is wasteful until we need it.
                    limit_mm_per_prompt={"image": 1, "video": 0},
                )
            self.sampling_params = SamplingParams(
                temperature=args.temperature,
                max_tokens=self.max_completion_length,
            )
            
        # determines when to update the model weights on vLLM.
        self._last_loaded_step = 0
            
        # When using vLLM, the main process is responsible for loading the model weights. This can cause process
        # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
        # synchronize all processes after vLLM has been fully initialized.
        self.accelerator.wait_for_everyone()
        
        self.env = env
        
        
        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False
        
        
        
        
        
        

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
