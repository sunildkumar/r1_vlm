from typing import Callable
import argparse
import re
import json
import os

from datasets import load_dataset, load_from_disk
from peft import LoraConfig
from prepare_inputs import tokenize_and_inject_images
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from trl import GRPOConfig, ModelConfig
from trl.trainer import QwenGRPOTrainer, ToolDefinition
import trl

from reward_fns import (
    answer_reward_func,
    bounding_box_presence_reward_func,
    format_reward_func,
    format_numeric_answer_reward_func,
    soft_answer_reward_func,
    tool_use_reward_func,
    magicword_reward_func,
)


class VerySimpleParser():

    def __init__(self):
        pass

    def extract_op_str(self, completion: str) -> str:
        """Extract the <op>...</op> string from the completion."""
        start = completion.find("<op>")
        end = completion.find("</op>")  
        if start == -1 or end == -1:
            # Don't include "<op>...</op>" in the response because that triggers the reward
            raise ValueError("Op command tag not found in completion")
        return completion[start+len("<op>"):end]


class ImageMetadataReadTool:
    """Tool that pretends to load an image, but actually reads the metadata next to the image.
    If return_nothing is True, the tool returns a newline character.  (Easy to mistake for real VLM output.)
    """
    def __init__(self, img_path: str, return_nothing: bool = False):
        self.img_path = os.path.expanduser(img_path)
        self.parser = VerySimpleParser()
        self.return_nothing = return_nothing

    def __call__(self, completion: str) -> str:
        op_str = self.parser.extract_op_str(completion)
        filename = self.parse_input(op_str)
        self.validate_filename(filename)
        data = self.read_metadata(filename)
        return self.format_response(data)
    
    def format_response(self, data: dict) -> str:
        """Format the response as a string."""
        if self.return_nothing:
            return "\n"
        else:
            j = json.dumps(data)
            return f"Image metadata: {j}"

    def parse_input(self, x: str) -> str:
        """Parse the input string to get the filename."""
        if not x.startswith("load: "):
            raise ValueError("Input does not start with load:")
        return x[len("load: "):]

    def validate_filename(self, filename: str):
        """Validate the filename.  Strict security check."""
        # We only allow filenames with alphanumeric and hyphen, and a single period.
        if not re.match(r"^[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-]+$", filename):
            raise ValueError("Invalid filename")
        if not filename.endswith(".jpeg"):
            raise ValueError("Invalid file extension")

    def read_metadata(self, filename: str) -> dict:
        """Read the metadata from the file."""
        # replace .jpeg with .json
        filename = filename.replace(".jpeg", ".json")
        path = os.path.join(self.img_path, filename)
        data = json.load(open(path))
        return data


def fake_tool(x: str) -> str:
    """Super stupid and useless tool that just demonstrates that
    the tool returns a string.
    """
    return "You think he's a tool!  What about me?"


class CompletelyEmptyToolDefinition(ToolDefinition):
    """A tool definition that never activates the tool.
    """

    def __init__(self):
        super().__init__(
            stop_string="</op>",
            call_tool=lambda x: x,
        )

    def completion_has_tool_call(self, completion: str) -> bool:
        return False


def parse_cli_args() -> argparse.Namespace:
    """
    Parses and returns command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train Qwen model with GRPOTrainer. Specify dataset load options."
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="cocomath",
        help="Name of the run",
    )
    parser.add_argument(
        "--rewards",
        type=str,
        default="cocomath",
        help="Which rewards to use",
    )
    parser.add_argument(
        "--tool_img_path",
        type=str,
        default="~/data/obfuscated_mnist",
        help="Path to the image to use for the tool",
    )
    parser.add_argument(
        "--which_tool",
        type=str,
        default="imgmetadata",
        help="Which tool to use",
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
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA for training",
    )
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=1024,
        help="Maximum completion length",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-7,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=50,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=5,
        help="Number of generations to use",
    )
    parser.add_argument(
        "--tool_return_nothing",
        action="store_true",
        help="Return nothing from the tool",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm",
    )
    parser.add_argument(
        "--loss_magnifier",
        type=float,
        default=1.0e4,
        help="Multiplies the loss on the way out to avoid underflow.",
    )
    return parser.parse_args()


def pick_rewards(args: argparse.Namespace) -> list[Callable]:
    if args.rewards == "cocomath":
        return [
            format_reward_func,
            answer_reward_func,
            soft_answer_reward_func,
            bounding_box_presence_reward_func,
        ]
    elif args.rewards == "mnist":
        return [
            #format_reward_func,  # Not sure this is working properly
            format_numeric_answer_reward_func,
            tool_use_reward_func,
            answer_reward_func,
        ]
    elif args.rewards == "toolformat":
        return [
            tool_use_reward_func,
            format_numeric_answer_reward_func,
        ]
    elif args.rewards == "justtool":
        return [
            tool_use_reward_func,
        ]
    elif args.rewards == "justformat":
        return [
            format_numeric_answer_reward_func,
        ]
    elif args.rewards == "magicword":
        return [
            magicword_reward_func,
        ]
    else:
        raise ValueError(f"Unknown rewards: {args.rewards}")

def main(args: argparse.Namespace):
    """
    Main training routine for Qwen model using GRPOTrainer.
    """
    if args.load_from_local:
        dataset = load_from_disk(args.local_path)
        print(f"Dataset loaded from local path: {args.local_path}")
    else:
        dataset = load_dataset("sunildkumar/coco-computation-r1")
        print("Dataset loaded from Hugging Face Hub.")

    # load the model
    model_config = ModelConfig(
        #model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
        model_name_or_path="Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype="bfloat16",
        use_peft=True,
    )
    #model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=model_config.model_name_or_path,
        torch_dtype=model_config.torch_dtype,
        # has to be set to false for gradient checkpointing to work
        use_cache=False,
        attn_implementation="flash_attention_2",
    )

    if args.use_lora:
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=4,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", padding_side="left"
    )

    # Hyperparameters
    training_args = GRPOConfig(
        output_dir=f"vlm-r1-{args.run_name}",
        learning_rate=args.learning_rate,
        adam_beta2=0.95,  # Faster adaptation to changes in variance
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        logging_steps=1,
        save_steps=100,
        save_total_limit=10,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        # GRPO specific parameters
        # TOOD: Make sure these are right
        max_prompt_length=1024,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        beta=0.001,  # KL penalty
        # TODO: True? using vllm seems like a good idea.
        use_vllm=False,
        report_to="wandb",
    )

    trainer = QwenGRPOTrainer(
        model=model,
        reward_funcs=pick_rewards(args),
        processing_class=processor,
        args=training_args,
        tokenize_and_inject_images=tokenize_and_inject_images,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        tool_defn=tool_defn_from_args(args),
    )
    trainer.train()


def tool_defn_from_args(args: argparse.Namespace) -> ToolDefinition | None:
    if args.which_tool == "none":
        return None
    if args.which_tool == "imgmetadata":
        tool = ImageMetadataReadTool(
            img_path=os.path.expanduser(args.tool_img_path),
            return_nothing=args.tool_return_nothing,
        )
        return ToolDefinition(
            stop_string="</op>",
            call_tool=tool,
        )
    if args.which_tool == "empty":
        return CompletelyEmptyToolDefinition()
    raise ValueError(f"Unknown tool: {args.which_tool}")

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
