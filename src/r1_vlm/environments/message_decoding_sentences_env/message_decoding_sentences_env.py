import re
from typing import Any, List

import Levenshtein
from datasets import Dataset, concatenate_datasets, load_dataset
from trl.trainer.grpo_trainer import RewardFunc
from verifiers.parsers import XMLParser

from r1_vlm.environments.simple_vision_env import SimpleVisionEnv


class MessageDecodingEnv(SimpleVisionEnv):
    def __init__(
        self,
        dataset: str = "sunildkumar/message-decoding-words-and-sentences-r1",
        system_prompt: str = "",
        **kwargs,  # passed to the superclass
    ):
        super().__init__(system_prompt=system_prompt, **kwargs)
        self.dataset_name = dataset
        self.parser = XMLParser(fields=["think", "answer"])

    def get_dataset(self) -> Dataset:
        dataset = load_dataset(self.dataset_name)["train"]

        # Curriculm learning
        # select 4000 "word" examples
        # select 4000 "word_pair" examples
        # then select all "sentence" examples, sorted from shortest to longest by "decoded_message" length (9k-ish)

        word_examples = dataset.filter(lambda x: x["task"] == "word")
        word_pair_examples = dataset.filter(lambda x: x["task"] == "word_pair")
        sentence_examples = dataset.filter(lambda x: x["task"] == "sentence")
        print(
            f"There are {len(word_examples)} word examples, {len(word_pair_examples)} word_pair examples, and {len(sentence_examples)} sentence examples"
        )

        # choose 4000 random "word" examples and word pair examples
        word_examples = word_examples.shuffle(seed=42).select(range(4000))

        word_pair_examples = word_pair_examples.shuffle(seed=42).select(range(4000))

        # sort sentence examples by length
        def add_length(example):
            example["length"] = len(example["decoded_message"])
            return example

        sentence_examples = sentence_examples.map(add_length)
        sentence_examples = sentence_examples.sort("length")
        sentence_examples = sentence_examples.remove_columns("length")

        # combine the three datasets in order.
        dataset = concatenate_datasets(
            [word_examples, word_pair_examples, sentence_examples]
        )

        return dataset

    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        def correctness_reward_func(completions, **kwargs) -> List[float]:
            """
            1.0 if exactly correct, otherwise 0.0
            """
            # parse the predicted decoded message from each completion
            responses = [self.parser.parse(c[0]["content"]).answer for c in completions]
            true_decoded_messages = kwargs["decoded_message"]

            def check_answer(response, answer):
                # the parser returns None if the answer is not found
                if response is None:
                    return 0.0

                try:
                    response = response.strip()
                    answer = answer.strip()
                except Exception as e:
                    print(f"Error in check_answer for correctness: {e}")
                    return 0.0

                if response == answer:
                    return 1.0
                else:
                    return 0.0

            rewards = [
                check_answer(r, t) for r, t in zip(responses, true_decoded_messages)
            ]
            return rewards

        def edit_distance_reward_func(completions, **kwargs) -> List[float]:
            """
            Reward proportional to the edit distance normalized by the length of the GT string and clamped between 0 and 1.

            Given GT and proposed string (PS):
                e_d = edit_distance(GT, PS)
                reward = 1 - e_d / max(len(GT), len(PS))
                reward = min(max(0.0, reward), 1.0)
            """

            # parse the predicted decoded message from each completion
            responses = [self.parser.parse(c[0]["content"]).answer for c in completions]
            true_decoded_messages = kwargs["decoded_message"]

            def check_answer(response, answer):
                # the parser returns None if the answer is not found
                if response is None:
                    return 0.0

                try:
                    response = response.strip()
                    answer = answer.strip()

                    edit_distance = Levenshtein.distance(response, answer)
                    reward = 1 - edit_distance / max(len(answer), len(response))
                    reward = min(max(0.0, reward), 1.0)
                    return reward
                except Exception as e:
                    print(f"Error in check_answer for edit distance: {e}")
                    return 0.0

            rewards = [
                check_answer(r, t) for r, t in zip(responses, true_decoded_messages)
            ]
            return rewards

        def check_format(text: str) -> float:
            """
            Helper function to check if the format is correct
            """
            # remove the bootstrap prompt from the text if it appears at the start
            text = text.removeprefix("Let me solve this step by step.\n")

            try:
                # Check if the format is correct
                regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"

                match = re.search(regex, text, re.DOTALL)

                if match is None or len(match.groups()) != 2:
                    return 0.0
                else:
                    return 1.0
            except Exception:
                return 0.0

        def format_reward_func(completions, **kwargs) -> List[float]:
            """
            Reward function that checks for proper XML formatting

            Must have think and answer fields
            """

            rewards = [check_format(c[0]["content"]) for c in completions]
            return rewards

        def thinking_reward(completions, **kwargs):
            """
            Rewards the model for thinking longer about the problem. Requires the model to achieve the format and correctness rewards
            in order to achieve this reward - that way we only reward thinking if it leads to a better model.
            """

            # find the completions that achieve this reward
            formatted_properly_list = format_reward_func(completions, **kwargs)
            answered_properly_list = correctness_reward_func(completions, **kwargs)

            can_achieve_thinking_reward = [
                f == 1.0 and a == 1.0
                for f, a in zip(formatted_properly_list, answered_properly_list)
            ]

            # if nothing can achieve the thinking reward, return 0s
            if not any(can_achieve_thinking_reward):
                return [0.0] * len(completions)

            # find how long each completion "thinks" for
            thinking_texts = [
                self.parser.parse(c[0]["content"]).think for c in completions
            ]
            thinking_texts = ["" if t is None else t for t in thinking_texts]
            thinking_lengths = [len(t) for t in thinking_texts]

            # find the max length among the thinking texts that can achieve the thinking reward
            max_len = 0
            for length, reward_available in zip(
                thinking_lengths, can_achieve_thinking_reward
            ):
                if reward_available:
                    max_len = max(max_len, length)

            # if no one thought, no thinking reward - e.g. <think></think> is the longest thinking text
            if max_len <= 0:
                return [0.0] * len(completions)

            # normalize the thinking lengths by the max length
            normalized_thinking_lengths = [
                length / max_len for length in thinking_lengths
            ]

            # a completion achieves the normalized reward if it meets the format and correctness requirements
            rewards = []
            for normalized_length, reward_available in zip(
                normalized_thinking_lengths, can_achieve_thinking_reward
            ):
                if reward_available:
                    rewards.append(normalized_length)
                else:
                    rewards.append(0.0)

            # these should all be between 0 and 1
            if max(rewards) > 1.0 or min(rewards) < 0.0:
                raise ValueError(f"Rewards are not between 0 and 1: {rewards}")

            return rewards

        # removed thinking_reward as it was actively hurting performance
        return [
            correctness_reward_func,
            edit_distance_reward_func,
            format_reward_func,
        ]
