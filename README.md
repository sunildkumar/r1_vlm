# r1_vlm
Trying GRPO on a toy task on a small VLM.


# Idea
This [blog post](https://www.philschmid.de/mini-deepseek-r1) shows how GRPO an LLM to do r1 style reasoning
on a toy problem. As far as I know, no one has tried this on a VLM. My idea is to generate a simple visual reasoning
dataset (not unlike the counting game in the blog post) and see if a VLM can do it.

## Dataset
Using the COCO dataset, I've generate a dataset of visual computation problems. For each image, I ask it to {add, subtract, multiply, divide}
the counts of two classes that are present in the image. For example, "Multiply the number of dogs by the number of cats in the image".

See the dataset [here](https://huggingface.co/datasets/sunildkumar/coco-computation-r1).


https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl is super helpful for sample code on how to FT a Qwen2VL generally. 

Custom fork of TRL for GRPO on VLMs: https://github.com/sunildkumar/trl. As of the time of writing, the latest version of GRPOTrainer does not support VLMs. 


## Training:
```
# run from root of repo
uv run accelerate launch --config_file train/multi_gpu.yaml  train/train.py

uv run accelerate launch --config_file train/multi_gpu.yaml  train/train_counting.py

CUDA_VISIBLE_DEVICES=1,2,3 uv run accelerate launch --config_file train/multi_gpu_3only.yaml  train/train_counting.py

CUDA_VISIBLE_DEVICES=1 uv run train/train_counting.py

CUDA_VISIBLE_DEVICES=1 uv run train/train_digit_recognition.py

CUDA_VISIBLE_DEVICES=1,2,3 uv run accelerate launch --config_file train/multi_gpu_3only.yaml train/train_digit_recognition.py 2>&1 | tee digit_recognition_logs_$(date +%Y%m%d_%H%M%S).log

CUDA_VISIBLE_DEVICES=1 uv run train/train_message_decoding.py

CUDA_VISIBLE_DEVICES=1,2,3 uv run accelerate launch --config_file train/multi_gpu_3only.yaml train/train_message_decoding.py 2>&1 | tee message_decoding_logs_$(date +%Y%m%d_%H%M%S).log


CUDA_VISIBLE_DEVICES=1 uv run train/train_digit_recognition_new_trainer.py

CUDA_VISIBLE_DEVICES=1,2,3 uv run accelerate launch --config_file train/multi_gpu_3only.yaml train/train_digit_recognition_new_trainer.py 2>&1 | tee digit_recognition_new_trainer_logs_$(date +%Y%m%d_%H%M%S).log

# 7b zero3
CUDA_VISIBLE_DEVICES=1,2,3 uv run accelerate launch --config_file train/multi_gpu_3only_zero3.yaml train/train_digit_recognition_new_trainer.py 2>&1 | tee digit_recognition_new_trainer_logs_$(date +%Y%m%d_%H%M%S).log

# 7b zero2 offload
CUDA_VISIBLE_DEVICES=1,2,3 uv run accelerate launch --config_file train/multi_gpu_3only_zero2_offload.yaml train/train_digit_recognition_new_trainer.py 2>&1 | tee digit_recognition_new_trainer_logs_$(date +%Y%m%d_%H%M%S).log

# 2b high completion run all gpu
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run accelerate launch --config_file train/multi_gpu.yaml train/train_digit_recognition_new_trainer.py 2>&1 | tee digit_recognition_new_trainer_logs_$(date +%Y%m%d_%H%M%S).log

# restart training for 2b high completion run all gpu but turn off cosine schedule
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run accelerate launch --config_file train/multi_gpu.yaml train/train_digit_recognition_new_trainer_no_schedule.py 2>&1 | tee digit_recognition_new_trainer_no_schedule_logs_$(date +%Y%m%d_%H%M%S).log


# 2b message decoding new trainer single gpu
CUDA_VISIBLE_DEVICES=1 uv run train/train_message_decoding_new_trainer.py

# 2b message decoding new trainer all gpu
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run accelerate launch --config_file train/multi_gpu.yaml train/train_message_decoding_new_trainer.py 2>&1 | tee message_decoding_new_trainer_logs_$(date +%Y%m%d_%H%M%S).log

# 3b word decoding single gpu
CUDA_VISIBLE_DEVICES=1 uv run train/train_message_decoding_words.py

# 3b word decoding 3 gpu
CUDA_VISIBLE_DEVICES=1,2,3 uv run accelerate launch --config_file train/multi_gpu_3only.yaml train/train_message_decoding_words.py 2>&1 | tee message_decoding_words_logs_$(date +%Y%m%d_%H%M%S).log



# 3b message decoding words vllm 3 gpu for train, 1 for vllm
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run accelerate launch --config_file train/multi_gpu_3only.yaml train/train_message_decoding_words_vllm.py 2>&1 | tee message_decoding_words_vllm_logs_$(date +%Y%m%d_%H%M%S).log


```

## Results
I just started model training on February 2nd, 2025 5:12:25 PM, and it's still only about 250 training steps in. Will update once I have results!
