import trl
from datasets import load_dataset
from peft import LoraConfig
from prepare_inputs import tokenize_and_inject_images
from digit_recognition_reward_fns import format_reward_func, answer_reward_func
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from trl import GRPOConfig, ModelConfig
from trl.trainer import QwenGRPOTrainer
from datasets import concatenate_datasets
from torch.utils.data import SequentialSampler
import math
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import matplotlib.pyplot as plt
import torch
from curriculum_utils import calculate_curriculum_steps, create_curriculum_lr_lambda, plot_lr_schedule
print(trl.__file__)


dataset = load_dataset("sunildkumar/digit-recognition-r1")
digits_1 = dataset["digits_1"].shuffle(seed=42)
digits_2 = dataset["digits_2"].shuffle(seed=42)
digits_3 = dataset["digits_3"].shuffle(seed=42)

# Split each dataset
split_1 = digits_1.train_test_split(test_size=0.1, seed=42)
split_2 = digits_2.train_test_split(test_size=0.1, seed=42)
split_3 = digits_3.train_test_split(test_size=0.1, seed=42)

digits_1_train, digits_1_eval = split_1["train"], split_1["test"]
digits_2_train, digits_2_eval = split_2["train"], split_2["test"]
digits_3_train, digits_3_eval = split_3["train"], split_3["test"]


train_dataset = concatenate_datasets([digits_1_train, digits_2_train, digits_3_train])
eval_dataset = concatenate_datasets([digits_1_eval, digits_2_eval, digits_3_eval])

print(f"There are {len(train_dataset)} training examples and {len(eval_dataset)} eval examples.")

# load the model
model_config = ModelConfig(
    model_name_or_path="Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype="bfloat16",
    use_peft=True,
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=model_config.model_name_or_path,
    torch_dtype=model_config.torch_dtype,
    # has to be set to false for gradient checkpointing to work
    use_cache=False,
    # faster generation, R1-V suggestion
    attn_implementation="flash_attention_2",
)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

# set the minimum pixels and maximum pixels to this value == 529 image tokens per image.
pixels = 224 * 224
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", padding_side="left", min_pixels=pixels, max_pixels=pixels
)

# Hyperparameters
training_args = GRPOConfig(
    output_dir="vlm-r1-digit-recognition",
    learning_rate=1e-6,
    lr_scheduler_type="cosine",
    warmup_steps=0,
    logging_steps=1,
    save_steps=20,
    # ckpts are 51 gb each!!
    save_total_limit= 50,
    num_train_epochs=1,
    # I've heard I shouldn't increase this due to a bug. 
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=False,
    bf16=True,
    # GRPO specific parameters
    # TOOD: Make sure these are right
    max_prompt_length=1024,
    max_completion_length=512,  # max length of the generated output for our solution
    num_generations=8,
    beta=0.001,
    use_vllm=False,
    report_to="wandb",
    # R1-V suggestion
    temperature=1.0, 
)

# Setup curriculum learning
dataset_sizes = [len(digits_1_train), len(digits_2_train), len(digits_3_train)]
num_gpus = torch.cuda.device_count()
transition_steps = calculate_curriculum_steps(
    dataset_sizes, 
    training_args.per_device_train_batch_size, 
    training_args.gradient_accumulation_steps,
    num_gpus
)
curriculum_lr_lambda = create_curriculum_lr_lambda(transition_steps)

# Visualize the schedule
plot_lr_schedule(transition_steps, curriculum_lr_lambda)

# turn off shuffling so the model sees the data in increasing difficulty order
class NoShuffleQwenGRPOTrainer(QwenGRPOTrainer):
    def __init__(self, *args, lr_lambda=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_lambda = lr_lambda

    def _get_train_sampler(self):
        return SequentialSampler(self.train_dataset)
    
    def create_scheduler(self, num_training_steps, optimizer=None):
        optimizer = self.optimizer if optimizer is None else optimizer
        self.lr_scheduler = LambdaLR(optimizer, self.lr_lambda)
        return self.lr_scheduler

trainer = NoShuffleQwenGRPOTrainer(
    model=model,
    reward_funcs=[
        format_reward_func,
        answer_reward_func,
    ],
    processing_class=processor,
    args=training_args,
    tokenize_and_inject_images=tokenize_and_inject_images,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # use our curriculum learning lambda for the lr scheduler
    lr_lambda=curriculum_lr_lambda,
    #    peft_config=peft_config,
)

trainer.train()
