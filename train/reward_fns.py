import re

# https://www.philschmid.de/mini-deepseek-r1#3-train-the-model-using-grpo-educational-part was the best reference for this
import numpy as np


def format_reward_func(completions, target, **kwargs):
    """
    Format: <think>...</think><answer>...</answer>
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers

      Returns:
          list[float]: Reward scores
    """
    rewards = []

    for completion_conv, gt in zip(completions, target):
        try:
            # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
            completion = "<think>" + completion_conv[0]["content"]

            # Check if the format is correct
            regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"

            match = re.search(regex, completion, re.DOTALL)

            # if the format is not correct, reward is 0
            if match is None or len(match.groups()) != 2:
                rewards.append(0.0)
            else:
                rewards.append(1.0)
        except Exception as e:
            print(f"Error in format_reward_func: {e}")
            rewards.append(0.0)

    return rewards


def format_numeric_answer_reward_func(completions, target, **kwargs):
    """
    Checks if the string inside <answer> parses as a float.
      Returns:
          list[float]: Reward scores
    """
    rewards = []

    for completion_conv, gt in zip(completions, target):
        try:
            # Look for <answer>...</answer>
            match = re.search(r"<answer>([\s\S]*?)<\/answer>", completion_conv[0]["content"])
            if match is None:
                rewards.append(0.0)
                continue

            answer = match.group(1).strip()
            try:
                float(answer)
                rewards.append(1.0)
            except ValueError:
                rewards.append(0.0)
        except Exception as e:
            print(f"Error in format_reward_func: {e}")
            rewards.append(0.0)

    return rewards


def tool_use_reward_func(completions, target, **kwargs):
    """
    Checks if the string inside <op>...</op> parses as a tool call.
    Returns:
        list[float]: Reward scores
    """
    rewards = []

    for completion_conv, gt in zip(completions, target):
        try:
            # Look for <op>...</op>
            match = re.search(r"<op>(.*?)<\/op>", completion_conv[0]["content"])
            if match is None:
                rewards.append(0.0)
            else:
                rewards.append(1.0)

        except Exception as e:
            print(f"Error in tool_use_reward_func: {e}")
            rewards.append(0.0)

    return rewards


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

            # check if float(answer) == gt (target is already a float)
            assert isinstance(gt, float), "target is not a float but a %s" % type(gt)

            # convert answer to float
            answer = float(answer)

            # check if the two are within e-5 of each other
            if abs(answer - gt) < 1e-5:
                rewards.append(1.0)
            else:
                rewards.append(0.0)

        except Exception as e:
            print(f"Error in answer_reward_func: {e}")
            rewards.append(0.0)

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

            # check if float(answer) == gt (target is already a float)
            assert isinstance(gt, float), "target is not a float but a %s" % type(gt)

            # convert answer to float
            answer = float(answer)

            error = abs(answer - gt)

            reward = np.exp(-1 * np.abs(error))
            rewards.append(reward)

        except Exception as e:
            print(f"Error in answer_reward_func: {e}")
            rewards.append(0.0)
    for prompt, completion_conv, gt, reward, class_1, class_2, count_1, count_2 in zip(
        kwargs["prompts"],
        completions,
        target,
        rewards,
        kwargs["class_1"],
        kwargs["class_2"],
        kwargs["count_1"],
        kwargs["count_2"],
    ):
        # Replace the entire sequence of padding tokens with a single placeholder
        prompt_cleaned = re.sub(
            r"(<\|image_pad\|>)+", "{... many image pad tokens ...}", prompt
        )
        print(f"Prompt: \n {prompt_cleaned}")
        print(f"Completion: \n {completion_conv[0]['content']}")
        print(f"Target: {gt}")
        print(f"Reward: {reward}")
        print(f"Class 1: {class_1}, {count_1}")
        print(f"Class 2: {class_2}, {count_2}")
        print("-" * 100)
    return rewards


def bounding_box_presence_reward_func(completions, target, **kwargs):
    """
    Evaluates completions based on the presence of bounding boxes or keypoints.

    Args:
        completions (list[str]): Generated outputs

    Returns:
        list[float]: Reward scores based on the count of detected bounding boxes and keypoints
    """
    rewards = []

    for completion_conv, gt, count1, count2 in zip(
        completions, target, kwargs.get("count_1", []), kwargs.get("count_2", [])
    ):
        try:
            completion = completion_conv[0]["content"]

            # Extract bounding boxes and keypoints from the completion
            bbox_matches = re.findall(
                r'{"bbox_2d": \[(.*?)\], "label": "(.*?)"}', completion
            )
            keypoint_matches = re.findall(
                r'{"point_2d": \[(.*?)\], "label": "(.*?)"}', completion
            )

            if count1 == 0 or count2 == 0:
                print(f"No count1 or count2 found in kwargs, {kwargs=}")

                rewards.append(0.0)
                continue

            # Count the number of bounding boxes and keypoints
            total_detected = len(bbox_matches) + len(keypoint_matches)

            # if we have fewer boxes than expected, we reward the model for each box it detected with 1/(count1 + count2)
            if total_detected <= count1 + count2:
                reward = total_detected / (count1 + count2)

            # if we have more boxes than expected, we give the model reward 1.0 minus 1/(count1 + count2) for each box over the expected count
            else:
                detection_overage = total_detected - (count1 + count2)
                penalty = detection_overage / (count1 + count2)
                reward = 1.0 - penalty

                # we still give a small reward if the model produces many more boxes than expected
                reward = max(0.1, reward)

            rewards.append(reward)
        except Exception as e:
            print(f"Error in bounding_box_presence_reward_func: {e}")
            rewards.append(0.0)

    return rewards


def test_reward_funcs():
    # simple correct example, int answer
    correct_sample_1 = [
        {
            "content": "We need to count the number of people in the image. There are 2 people in the image. "
            "We need to count the number of bananas in the image. There are 4 bananas in the image. "
            "We need to multiply the number of people by the number of bananas. 2 * 4 = 8. </think>\n"
            "<answer> 8 </answer>"
        }
    ]

    # correct example, float answer
    correct_sample_2 = [
        {
            "content": "We need to count the number of people in the image. There are 2 people in the image. "
            "We need to count the number of bananas in the image. There are 4 bananas in the image. "
            "We need to multiply the number of people by the number of bananas. 2 * 4 = 8.0 </think>\n"
            "<answer> 8.0 </answer>"
        }
    ]

    # correct example, division needed, truncation needed
    correct_sample_3 = [
        {
            "content": "We need to count the number of people in the image. There are 1 people in the image. "
            "We need to count the number of bananas in the image. There are 3 bananas in the image. "
            "We need to divide the number of people by the number of bananas. 1 / 3 = 0.3333333333333333.</think>\n"
            "<answer> 0.33 </answer>"
        }
    ]

    # correct example, no thinking
    correct_sample_4 = [{"content": "</think>\n<answer> 8 </answer>"}]

    # incorrect format, no end of thinking, no start of answer.
    wrong_format_1 = [{"content": "The answer is 8"}]

    # multiple thinking sections
    wrong_format_2 = [
        {
            "content": "We need to count the number of people in the image. There are 1 people in the image. </think> "
            "<think>We need to count the number of bananas in the image. There are 3 bananas in the image. </think> "
            "<think>We need to divide the number of people by the number of bananas. 1 / 3 = 0.3333333333333333.</think> "
            "<answer> 0.33 </answer>"
        }
    ]

    # correct format (no thinking), wrong result
    wrong_result_1 = [{"content": "</think><answer> 6 </answer>"}]

    test_rewards_format = format_reward_func(
        completions=[
            correct_sample_1,
            correct_sample_2,
            correct_sample_3,
            correct_sample_4,
            wrong_format_1,
            wrong_format_2,
            wrong_result_1,
        ],
        target=[8.0, 8.0, 0.33, 8.0, 8.0, 8.0, 8.0],
    )
    assert test_rewards_format == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], (
        "format reward func not working properly"
    )

    test_rewards_answer = answer_reward_func(
        completions=[
            correct_sample_1,
            correct_sample_2,
            correct_sample_3,
            correct_sample_4,
            wrong_format_1,
            wrong_format_2,
            wrong_result_1,
        ],
        target=[8.0, 8.0, 0.33, 8.0, 8.0, 8.0, 8.0],
    )

    assert test_rewards_answer == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], (
        "answer reward func not working properly"
    )


def test_bounding_box_presence_reward_func():
    # Test case where count1 = count2 = 1, and the completion has 1 bbox and 1 keypoint
    completion = [
        {
            "content": '{"bbox_2d": [10, 20, 30, 40], "label": "object1"} '
            '{"point_2d": [15, 25], "label": "keypoint1"}'
        }
    ]

    kwargs = {"count_1": 1, "count_2": 1}

    rewards = bounding_box_presence_reward_func(
        completions=[completion], target=[None], **kwargs
    )

    expected_reward = (
        1.0  # Since total_detected = 2 and count1 + count2 = 2, reward should be 1.0
    )
    assert rewards == [expected_reward], (
        f"Expected {expected_reward}, but got {rewards[0]}"
    )

    kwargs = {"count_1": 2, "count_2": 1}

    rewards = bounding_box_presence_reward_func(
        completions=[completion], target=[None], **kwargs
    )

    expected_reward = 2 / 3
    assert rewards == [expected_reward], (
        f"Expected {expected_reward}, but got {rewards[0]}"
    )

    completion = [
        {
            "content": '{"bbox_2d": [10, 20, 30, 40], "label": "object1"} '
            '{"point_2d": [15, 25], "label": "keypoint1"} '
            '{"point_2d": [15, 25], "label": "keypoint1"} '
            '{"point_2d": [15, 25], "label": "keypoint1"} '
        }
    ]

    kwargs = {"count_1": 1, "count_2": 1}
    rewards = bounding_box_presence_reward_func(
        completions=[completion], target=[None], **kwargs
    )

    expected_reward = 0.1
    assert rewards == [expected_reward], (
        f"Expected {expected_reward}, but got {rewards[0]}"
    )


if __name__ == "__main__":
    test_reward_funcs()
    test_bounding_box_presence_reward_func()
