# Implementation provenance

Method: Geoffrey Hinton, Oriol Vinyals, Jeff Dean, *Distilling the Knowledge in a
Neural Network*, 2015, https://arxiv.org/abs/1503.02531 .

Model and distillation implementation reference: PyTorch Knowledge Distillation
Tutorial by Alexandros Chariton, from the pytorch/tutorials repository, commit
c8ccbbf7fcb8cb71adda1b957bd6bab50e29c2e6. DeepNN and LightNN class source is
copied verbatim into vendor/pytorch_tutorial/models.py. The original source,
BSD-3-Clause license and byte hashes are retained alongside it. The full original
tutorial is stored as reference_source.txt and is never executed on import.

The KD loss uses torch.nn.functional.kl_div for numerically stable zero-probability
handling, mathematically equivalent to the tutorial's teacher-to-student KL
expression including the T² factor. Data partitioning, independent validation
selection, paired repeats, Windows entry points, checkpoint recovery, evaluation
and aggregation adapt the tutorial to the present controlled experiment.

Atomic checkpoint, RNG restoration, verified stratified-split, data preparation
and hash-tracking utilities reuse code from the previously prepared Cutout course
experiment scaffold. No Cutout model, training weight, algorithm or result is
used in this new study. Existing CIFAR-10 files may be copied after official
integrity checks. This is a core-method reproduction on CIFAR-10, not a claim
to reproduce every MNIST/speech/specialist experiment in the original paper.
