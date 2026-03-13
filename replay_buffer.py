"""
Replay buffer for GAN training.

Stores previously generated images to stabilize discriminator updates.
"""

import random
import torch


class ReplayBuffer:
    """
    Fixed-size buffer that returns a mix of old and new fake samples.

    This helps reduce model oscillation during adversarial training.
    """

    def __init__(self, max_size=50):
        """
        Args:
            max_size (int): Maximum number of stored tensors.
        """
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, tensors):
        """
        Push new tensors and return a batch that mixes old and new samples.

        Args:
            tensors (torch.Tensor): Batch of generated images.

        Returns:
            torch.Tensor: Batch with some images replaced by buffer samples.
        """
        output = []
        for tensor in tensors.detach():
            tensor = tensor.unsqueeze(0)
            if len(self.data) < self.max_size:
                # Buffer not full yet: store and return current tensor.
                self.data.append(tensor)
                output.append(tensor)
            else:
                # With 50% probability, return a past sample to stabilize training.
                if random.random() > 0.5:
                    idx = random.randint(0, self.max_size - 1)
                    output.append(self.data[idx].clone())
                    self.data[idx] = tensor
                else:
                    output.append(tensor)
        return torch.cat(output)
