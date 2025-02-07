import numpy as np
import math
import matplotlib.pyplot as plt

def calculate_curriculum_steps(dataset_sizes, batch_size, gradient_accumulation, num_gpus=1):
    """Convert dataset sizes into training step counts for curriculum transitions.
    
    For curriculum learning with multiple datasets of increasing difficulty, this function
    calculates at which training step to transition from one dataset to the next.
    
    Args:
        dataset_sizes (List[int]): Number of samples in each dataset of increasing difficulty
        batch_size (int): Number of samples processed per device per forward pass
        gradient_accumulation (int): Number of forward passes before gradient update
        num_gpus (int, optional): Number of GPUs for distributed training. Defaults to 1
        
    Returns:
        np.ndarray: Array of step counts [0, s1, s2, ..., sn] where:
            - 0 is the start
            - s1 is when to transition from dataset 1 to 2
            - s2 is when to transition from dataset 2 to 3
            - sn is the total number of steps
    """
    effective_batch_size = batch_size * gradient_accumulation * num_gpus
    return np.cumsum([0] + [
        size // effective_batch_size 
        for size in dataset_sizes
    ])

def create_curriculum_lr_lambda(transition_steps):
    """Create learning rate scheduler that resets and decays within each curriculum stage.
    
    Implements cosine decay that resets at the start of each dataset in the curriculum.
    The learning rate follows cos(πx/2)² decay within each stage, starting at 1.0
    and decaying to a minimum before the next stage begins.
    
    Args:
        transition_steps (np.ndarray): Array of step counts marking dataset transitions,
            as returned by calculate_curriculum_steps()
    
    Returns:
        Callable[[int], float]: Function that takes the current training step
            and returns a learning rate multiplier between 0 and 1
    """
    def lr_lambda(current_step):
        dataset_idx = np.searchsorted(transition_steps[1:], current_step, side='right')
        steps_in_current_dataset = current_step - transition_steps[dataset_idx]
        segment_length = transition_steps[dataset_idx + 1] - transition_steps[dataset_idx]
        return math.cos(math.pi * steps_in_current_dataset / segment_length / 2) ** 2
    return lr_lambda

def plot_lr_schedule(transition_steps, lr_lambda):
    """Visualize the learning rate schedule."""
    total_steps = transition_steps[-1]
    steps = range(total_steps)
    lr_multipliers = [lr_lambda(step) for step in steps]

    plt.figure(figsize=(12, 6))
    plt.plot(steps, lr_multipliers)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate Multiplier')

    for step in transition_steps[1:-1]:
        plt.axvline(x=step, color='r', linestyle='--', alpha=0.3)

    plt.grid(True)
    plt.savefig('lr_schedule.png') 