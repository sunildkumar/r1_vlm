# training_config.py
from transformers import TrainingArguments
from trl import PPOTrainer, PPOConfig

vision_ppo_config = PPOConfig(
    model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    learning_rate=1.4e-5,
    batch_size=4,
    ppo_epochs=3,
    gradient_accumulation_steps=2,
    optimize_cuda_cache=True,
    tokenizer_name="Qwen/Qwen2.5-VL-3B-Instruct",
    tracker_project_name="vl_tool_training"
)

training_args = TrainingArguments(
    output_dir="./vl_tool_trainer",
    remove_unused_columns=False,
    report_to="wandb",
    logging_steps=10,
)

