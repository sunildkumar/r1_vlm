import os

from digit_recognition_env import DigitRecognitionEnv
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import GRPOConfig, ModelConfig
from trl.trainer.qwen_grpo_trainer import QwenGRPOTrainer

os.environ["WANDB_ENTITY"] = "groundlightai"
os.environ["WANDB_PROJECT"] = "digit-recognition-verifiers-integration"


vf_env = DigitRecognitionEnv()
dataset = vf_env.get_dataset()
rubric = vf_env.get_rubric()


# Flag that determines if gradient checkpointing is used. If it is, we need to set use_cache to False.
gradient_checkpointing = False


model_config = ModelConfig(
    model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype="bfloat16",
    use_peft=False,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=model_config.model_name_or_path,
    torch_dtype=model_config.torch_dtype,
    use_cache=False,
    # flash attention not supported on our trainer yet
    # attn_implementation="flash_attention_2",
)

# use cache if not gradient checkpointing
if gradient_checkpointing:
    model.config.use_cache = False
elif not gradient_checkpointing:
    model.config.use_cache = True
else:
    raise ValueError("Invalid gradient checkpointing value")


processor = AutoProcessor.from_pretrained(
    model_config.model_name_or_path, padding_side="left"
)

# model on gpu 0, vllm on gpu 1

training_args = GRPOConfig(
    model_init_kwargs=model_config,
    output_dir="vlm-r1-digit-recognition-verifiers-integration",
    learning_rate=1e-6,
    adam_beta2=0.98,
    lr_scheduler_type="cosine",
    warmup_steps=0,
    logging_steps=1,
    save_steps=20,
    save_total_limit=50,
    num_train_epochs=1,
    # starting with 1 gpu and small number of completions
    per_device_train_batch_size=2,
    num_generations=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=gradient_checkpointing,
    bf16=True,
    # GRPO specific parameters
    max_prompt_length=None,  # must be None for vllm + verifiers
    max_completion_length=512,
    beta=0.001,
    temperature=1.0,
    sync_ref_model=True,
    ref_model_sync_steps=64,
    eval_strategy="no",
    log_completions=True,
    use_vllm=True,
    vllm_gpu_memory_utilization=0.5,
    report_to="wandb",
    vllm_device="cuda:1",
)


trainer = QwenGRPOTrainer(
    model=model,
    processing_class=processor,
    reward_funcs=rubric,
    tokenize_and_inject_images=None,
    args=training_args,
    train_dataset=dataset,
    env=vf_env,
)

# trainer.train()
