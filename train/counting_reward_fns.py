import os
import re

import numpy as np

# https://www.philschmid.de/mini-deepseek-r1#3-train-the-model-using-grpo-educational-part was the best reference for this

SHOULD_PRINT_REWARD = os.environ.get("SHOULD_PRINT_REWARD", "true").lower() == "true"


def print_reward(
    function_name,
    prompts,
    completions,
    target,
    rewards,
    additional_fields,
    reward_function_kwargs,
):
    """
    Prints reward function results with arbitrary additional fields.

    Args:
        function_name (str): Name of the reward function being used
        prompts (list): List of input prompts
        completions (list): List of model completions
        target (list): List of target values
        rewards (list): List of reward values
        additional_fields (list[str]): List of additional field names to print from kwargs
        reward_function_kwargs (dict): Kwargs containing additional data fields
    """
    if not SHOULD_PRINT_REWARD:
        return

    print(f"\nExecuting {function_name}")
    print("=" * 100)

    for idx, (prompt, completion_conv, gt, reward) in enumerate(
        zip(prompts, completions, target, rewards)
    ):
        # Clean up image padding tokens
        prompt_cleaned = re.sub(
            r"(<\|image_pad\|>)+", "{... many image pad tokens ...}", prompt
        )
        print(f"Function name: {function_name}")
        print(f"Sample {idx + 1}:")
        print(f"Prompt:\n{prompt_cleaned}")
        print(f"Completion:\n{completion_conv[0]['content']}")
        print(f"Target: {gt}")
        print(f"Reward: {reward}")

        # Print any additional fields requested
        for field in additional_fields:
            if field in reward_function_kwargs:
                value = reward_function_kwargs[field][idx]
                print(f"{field}: {value}")
            else:
                print(f"Can't find {field} in reward_function_kwargs")

        print("-" * 100)


def answer_reward_func(completions, target, **kwargs):
    """
    Evaluates completions based on mathematical correctness of the answer

    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
        nums (list[str]): Available numbers

    Returns:
        list[float]: Reward scores
    """
    SHOULD_PRINT_REWARD = True
    rewards = []

    for completion_conv, gt in zip(completions, target):
        if type(gt) != int:
            raise ValueError(f"Target is not an int: {gt}! This is unexpected.")
        
        try:
            # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
            completion = "<think>" + completion_conv[0]["content"]

            # Check if the format is correct
            match = re.search(r"<answer>(.*?)<\/answer>", completion)

            if match is None:
                rewards.append(0.0)
                continue

            answer = match.group(1).strip()

            # Define a regex pattern that only allows numbers, period, and whitespace
            allowed_pattern = r"^[\d\s.]+$"
            if not re.match(allowed_pattern, answer):
                rewards.append(0.0)
                continue
            
            
            # convert to floats for comparison
            gt = float(gt)

            answer = float(answer)

            # check if the two are within e-5 of each other
            if abs(answer - gt) < 1e-5:
                rewards.append(1.0)
            else:
                rewards.append(0.0)

        except Exception as e:
            print(f"Error in answer_reward_func: {e}")
            rewards.append(0.0)

    if SHOULD_PRINT_REWARD:
        print_reward(
            "answer_reward_func",
            kwargs.get("prompts", []),
            completions,
            target,
            rewards,
            ["class_1"],
            kwargs,
        )

    return rewards


def soft_answer_reward_func(completions, target, **kwargs):
    """
    Evaluates completions based on mathematical correctness of the answer.
    Uses an exponential function to reward correct answers, and penalize incorrect answers based on the error between the answer and the target.

    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
        nums (list[str]): Available numbers

    Returns:
        list[float]: Reward scores
    """
    SHOULD_PRINT_REWARD = True
    rewards = []

    for completion_conv, gt in zip(completions, target):
        try:
            # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
            completion = "<think>" + completion_conv[0]["content"]

            # Check if the format is correct
            match = re.search(r"<answer>(.*?)<\/answer>", completion)

            if match is None:
                rewards.append(0.0)
                continue

            answer = match.group(1).strip()

            # Define a regex pattern that only allows numbers, period, and whitespace
            allowed_pattern = r"^[\d\s.]+$"
            if not re.match(allowed_pattern, answer):
                rewards.append(0.0)
                continue

            assert isinstance(gt, int), "target is not an int but a %s" % type(gt)
            gt = float(gt)

            # convert answer to float
            answer = float(answer)

            error = abs(answer - gt)

            sigma = 2
            exponent = -1 * (error**2)
            exponent = exponent / (sigma**2)

            reward = np.exp(exponent)
            # convert from np.float64 to python float
            reward = float(reward)

            rewards.append(reward)

        except Exception as e:
            print(f"Error in answer_reward_func: {e}")
            rewards.append(0.0)
            
    if SHOULD_PRINT_REWARD:
        print_reward(
            "soft_answer_reward_func",
            kwargs.get("prompts", []),
            completions,
            target,
            rewards,
            ["class_1"],
            kwargs,
        )

    return rewards


def bounding_box_reward_func(completions, target, **kwargs):
    """
    Gives a reward for each bounding box/keypoint in the completion.
    Reward is min(1, total boxes+keypoints detected / (target))

    Args:
        completions (list[str]): Generated outputs

    Returns:
        list[float]: Reward scores based on the count of detected bounding boxes and keypoints
        
    See https://www.desmos.com/calculator/zt4mng4dst for plot of reward function
    """
    SHOULD_PRINT_REWARD = True
    
    # multiply the reward by this factor. Otherwise the reward isn't motivating enough. 
    REWARD_FACTOR = 30 
    rewards = []

    for completion_conv, gt in zip(completions, target):
        try:
            completion = "<think>" + completion_conv[0]["content"]

            # Extract content between think tags
            think_match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
            if not think_match:
                rewards.append(0.0)
                continue

            think_content = think_match.group(1)

            # Extract bounding boxes and keypoints from the think content
            bbox_matches = re.findall(
                r'{"bbox_2d": \[(.*?)\], "label": "(.*?)"}', think_content
            )
            keypoint_matches = re.findall(
                r'{"point_2d": \[(.*?)\], "label": "(.*?)"}', think_content
            )

            total_detected = len(bbox_matches) + len(keypoint_matches)
            
            if total_detected <= gt:
                reward = total_detected / gt
            else:
                # Exponential decay for excess detections
                excess = total_detected - gt
                reward = np.exp(-excess / gt)
            
            reward = reward * REWARD_FACTOR
            
            rewards.append(reward)
        except Exception as e:
            print(f"Error in bounding_box_reward_func: {e}")
            rewards.append(0.0)

    if SHOULD_PRINT_REWARD:
        print_reward(
            "bounding_box_reward_func",
            kwargs.get("prompts", []),
            completions,
            target,
            rewards,
            ["class_1"],
            kwargs,
        )

    return rewards
