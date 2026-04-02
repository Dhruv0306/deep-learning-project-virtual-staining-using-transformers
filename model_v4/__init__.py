"""model_v4 package — CUT + Transformer Phase-based implementation.

Current: unpaired GAN with PatchNCE contrastive loss and a Transformer
encoder. The ResNet generator is still available for ablations.

Public API:
    ResnetGenerator        — CNN generator
    PatchGANDiscriminator  — 70×70 PatchGAN discriminator
    getGeneratorV4         — factory + weight init for the generator
    getDiscriminatorV4     — factory + weight init for the discriminator
    train_v4               — training loop (LSGAN + PatchNCE + Transformer)
"""

from .generator import ResnetGenerator, TransformerGeneratorV4, getGeneratorV4
from .discriminator import PatchGANDiscriminator, getDiscriminatorV4
from .patch_sampler import PatchSampler
from .nce_loss import PatchNCELoss
from .training_loop import train_v4

__all__ = [
    "ResnetGenerator",
    "TransformerGeneratorV4",
    "PatchGANDiscriminator",
    "getGeneratorV4",
    "getDiscriminatorV4",
    "PatchSampler",
    "PatchNCELoss",
    "train_v4",
]
