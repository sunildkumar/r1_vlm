# training_config.py
from transformers import TrainingArguments
from trl import PPOTrainer, PPOConfig

vision_ppo_config = PPOConfig(
    model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    learning_rate=2e-3,
    batch_size=16,
    ppo_epochs=10,
    gradient_accumulation_steps=2,
    optimize_cuda_cache=True,
    tokenizer_name="Qwen/Qwen2.5-VL-3B-Instruct",
    tracker_project_name="mnist_tool",
)

training_args = TrainingArguments(
    output_dir="./output/mnist_tool",
    remove_unused_columns=False,
    report_to="wandb",
    logging_steps=10,
)

