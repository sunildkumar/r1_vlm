import trl
from datasets import load_dataset, load_from_disk
import argparse
from peft import LoraConfig
from prepare_inputs import tokenize_and_inject_images
from reward_fns import (
    answer_reward_func,
    bounding_box_presence_reward_func,
    format_reward_func,
    soft_answer_reward_func,
)
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import GRPOConfig, ModelConfig
from trl.trainer import QwenGRPOTrainer, ToolDefinition

print(trl.__file__)

def fake_tool(x: str) -> str:
    import pdb; pdb.set_trace()
    return "You think he's a tool!  What about me?"

def parse_cli_args() -> argparse.Namespace:
    """
    Parses and returns command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train Qwen2.5-VL model with GRPOTrainer. Specify dataset load options."
    )
    parser.add_argument(
        "--load_from_local",
        action="store_true",
        help="Load dataset from local disk instead of from HF Hub",
    )
    parser.add_argument(
        "--local_path",
        type=str,
        default="processed_r1_dataset",
        help="Local directory to load the dataset from",
    )
    return parser.parse_args()

def main(args: argparse.Namespace):
    """
    Main training routine for Qwen2.5-VL model using GRPOTrainer.
    """
    if args.load_from_local:
        dataset = load_from_disk(args.local_path)
        print(f"Dataset loaded from local path: {args.local_path}")
    else:
        dataset = load_dataset("sunildkumar/coco-computation-r1")
        print("Dataset loaded from Hugging Face Hub.")

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
        r=4,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", padding_side="left"
    )

    # Hyperparameters
    training_args = GRPOConfig(
        output_dir="vlm-r1-aha-moment",
        learning_rate=5e-7,
        lr_scheduler_type="cosine",
        warmup_ratio=0.001,  #  1M examples * 0.001 = 1000 steps
        logging_steps=1,
        save_steps=1,
        # roughly 1M total training steps
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        # GRPO specific parameters
        # TOOD: Make sure these are right
        max_prompt_length=1024,
        max_completion_length=1024,  # max length of the generated output for our solution
        num_generations=3,
        beta=0.001,
        # TODO: True? using vllm seems like a good idea.
        use_vllm=False,
        report_to="wandb",
    )

    trainer = QwenGRPOTrainer(
        model=model,
        reward_funcs=[
            format_reward_func,
            answer_reward_func,
            soft_answer_reward_func,
            bounding_box_presence_reward_func,
        ],
        processing_class=processor,
        args=training_args,
        tokenize_and_inject_images=tokenize_and_inject_images,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        #    peft_config=peft_config,
        tool_defn=ToolDefinition(
            stop_string="</op>",
            call_tool=fake_tool,
        ),
    )

    trainer.train()

if __name__ == "__main__":
    args = parse_cli_args()
    main(args)

# TODOs for Trainer
# 1. [x] Only support passing the model not a model name, it's too much to support both
# 2. [x]  Only support passing the processing class directly, don't support using their AutoTokenizer stuff.
# 3. [x]  what the heck is reward_processing_classes - not relevant as we aren't using model as reward function
# 4. []  Not worrying about the vllm stuff for now.
# 5. []  What temperature are they using by default?
# 6. [x]  Not worrying about deepspeed/multiple GPUs for now - multi gpu now supported
# 7. [x]  Update compute_loss to do tokenization/collating, maybe we give the trainer a function that is called there.
