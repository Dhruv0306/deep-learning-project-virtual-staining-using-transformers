"""model_v4 package — CUT + Transformer Phase-based implementation.

Phase 1 (this package): baseline unpaired GAN with a ResNet generator and
a PatchGAN discriminator.  Later phases will add PatchNCE contrastive loss
(Phase 2) and a Transformer encoder (Phase 3).

Public API:
    ResnetGenerator        — Phase 1 CNN generator
    PatchGANDiscriminator  — Phase 1 70×70 PatchGAN discriminator
    getGeneratorV4         — factory + weight init for the generator
    getDiscriminatorV4     — factory + weight init for the discriminator
    train_v4_phase1        — Phase 1 LSGAN training loop
"""

from .generator import ResnetGenerator, getGeneratorV4
from .discriminator import PatchGANDiscriminator, getDiscriminatorV4
from .training_loop import train_v4_phase1

__all__ = [
    "ResnetGenerator",
    "PatchGANDiscriminator",
    "getGeneratorV4",
    "getDiscriminatorV4",
    "train_v4_phase1",
]
