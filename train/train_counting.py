import trl
from counting_reward_fns import (
    answer_reward_func,
    bounding_box_reward_func,
    soft_answer_reward_func,
)
from datasets import load_dataset
from peft import LoraConfig
from prepare_inputs import tokenize_and_inject_images
from reward_fns import format_reward_func
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import GRPOConfig, ModelConfig
from trl.trainer import QwenGRPOTrainer

print(trl.__file__)


# 100k-ish examples
dataset = load_dataset("sunildkumar/coco-counts-balanced-r1")


# load the model
model_config = ModelConfig(
    model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype="bfloat16",
    use_peft=True,
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=model_config.model_name_or_path,
    torch_dtype=model_config.torch_dtype,
    # has to be set to false for gradient checkpointing to work
    use_cache=False,
)
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", padding_side="left"
)


# Hyperparameters
training_args = GRPOConfig(
    output_dir="vlm-r1-aha-moment-counting-only",
    learning_rate=5e-7,
    lr_scheduler_type="cosine",
    warmup_steps=25,
    logging_steps=1,
    save_steps=20,
    # ckpts are 51 gb each!!
    save_total_limit= 50,

    num_train_epochs=1,
    per_device_train_batch_size=1,
    # TODO: increase this to 4
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=True,
    # GRPO specific parameters
    # TOOD: Make sure these are right
    max_prompt_length=1024,
    max_completion_length=1024,  # max length of the generated output for our solution
    num_generations=5,
    beta=0.001,
    # TODO: True? using vllm seems like a good idea.
    use_vllm=False,
    report_to="wandb",
)


trainer = QwenGRPOTrainer(
    model=model,
    reward_funcs=[
        answer_reward_func,
        soft_answer_reward_func,
        bounding_box_reward_func,
        format_reward_func,
    ],
    processing_class=processor,
    args=training_args,
    tokenize_and_inject_images=tokenize_and_inject_images,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    #    peft_config=peft_config,
)

trainer.train()
