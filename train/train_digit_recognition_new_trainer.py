import os

import torch
import trl
from curriculum_utils import (
    calculate_curriculum_steps,
    create_curriculum_lr_lambda,
    plot_lr_schedule,
)
from datasets import concatenate_datasets, load_dataset
from digit_recognition_reward_fns import answer_reward_func, format_reward_func
from prepare_inputs import tokenize_and_inject_images
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from trl import GRPOConfig, ModelConfig
from trl.trainer.qwen_grpo_trainer import QwenGRPOTrainer

os.environ["WANDB_ENTITY"] = "groundlightai"
os.environ["WANDB_PROJECT"] = "digit-recognition-new-trainer"

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

print(
    f"There are {len(train_dataset)} training examples and {len(eval_dataset)} eval examples."
)

# Flag that determines if gradient checkpointing is used. If it is, we need to set use_cache to False.
gradient_checkpointing = False

model_config = ModelConfig(
    model_name_or_path="Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype="bfloat16",
    use_peft=False,
    #    attn_implementation="flash_attention_2",
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=model_config.model_name_or_path,
    torch_dtype=model_config.torch_dtype,
    # faster generation, R1-V suggestion
    #    attn_implementation=model_config.attn_implementation,
)

# use cache if not gradient checkpointing
if gradient_checkpointing:
    model.config.use_cache = False
elif not gradient_checkpointing:
    model.config.use_cache = True
else:
    raise ValueError("Invalid gradient checkpointing value")

pixels = 224 * 224
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    padding_side="left",
    min_pixels=pixels,
    max_pixels=pixels,
)


training_args = GRPOConfig(
    model_init_kwargs=model_config,
    output_dir="vlm-r1-digit-recognition",
    learning_rate=1e-6,
    # reduce beta2 as our number of steps is small
    adam_beta2=0.98,
    lr_scheduler_type="cosine",
    warmup_steps=0,
    logging_steps=1,
    save_steps=20,
    # ckpts are 51 gb each!!
    save_total_limit=50,
    num_train_epochs=1,
    # represents the number of generations per device
    per_device_train_batch_size=10,
    # number of generations total - should be per_device_train_batch_size * num_gpus
    num_generations=40,
    gradient_accumulation_steps=4,
    gradient_checkpointing=gradient_checkpointing,
    bf16=True,
    # GRPO specific parameters
    max_prompt_length=1024,
    max_completion_length=512,  # max length of the generated output for our solution
    beta=0.001,
    use_vllm=False,
    report_to="wandb",
    # R1-V suggestion
    temperature=1.0,
    # sync the reference model every so often
    sync_ref_model=True,
    # how often to merge the reference model with the train model, default is 64
    ref_model_sync_steps=64,
    eval_strategy="no",
    log_completions=True,
)

# Setup curriculum learning
dataset_sizes = [len(digits_1_train), len(digits_2_train), len(digits_3_train)]
num_gpus = torch.cuda.device_count()
transition_steps = calculate_curriculum_steps(
    dataset_sizes,
    1,  # the per device batch size - manually setting this here because the value in args is not what it seems
    training_args.gradient_accumulation_steps,
    1,  # the number of GPUs isn't relevant here (I think??) because of the change to generation setup.
)
print(f"Transition steps: {transition_steps}")
curriculum_lr_lambda = create_curriculum_lr_lambda(transition_steps)
plot_lr_schedule(transition_steps, curriculum_lr_lambda)


# Customize the trainer to use our curriculum learning lambda for the lr scheduler
class LRLambdaQwenGRPOTrainer(QwenGRPOTrainer):
    def __init__(self, *args, lr_lambda=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_lambda = lr_lambda

    def create_scheduler(self, num_training_steps, optimizer=None):
        optimizer = self.optimizer if optimizer is None else optimizer
        self.lr_scheduler = LambdaLR(optimizer, self.lr_lambda)
        return self.lr_scheduler


trainer = LRLambdaQwenGRPOTrainer(
    model=model,
    processing_class=processor,
    reward_funcs=[
        format_reward_func,
        answer_reward_func,
    ],
    tokenize_and_inject_images=tokenize_and_inject_images,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # Don't shuffle the dataset so we train on the curriculm in order - 1 digit, then 2 digits, then 3 digits.
    shuffle_dataset=False,
    # use our curriculum learning lambda for the lr scheduler
    lr_lambda=curriculum_lr_lambda,
)

trainer.train()
# TODOS:
# [x] - the section per_device_train/eval_batch_size * num processes can be divided by the number of generations seems a bit worrying. It might limit num_generations to 4?
# [x] - inject images into the input at the top of _prepare_inputs
# [] - add logging metrics suggested by tyler
# [] -

# GOALS:
# [DONE] - Branch off of TRL main (again) with the new version of their trainer. Get 2B training off of it on digits on a single GPU.
# [DONE] - Get 2B training off of it on digits on multiple GPUs - zero2.
# [DONE] - Get 7B model training on digits task - maybe zero3 if necessary? Prove we can solve it to prove the code works. I can do it on zero2 offload optimizer and params.
# [TODO] - Get VLLM working. It should make the generation step a whole lot faster and thus training faster.
# [TODO] - Try the decoding task again.

# NOTES:
