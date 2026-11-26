# TODO

## Population and reproduction
- Support reproduction when genera have unequal numbers of `NeatModule` instances or incompatible parameter shapes.
- Implement `Population.reset_keys()` and define how genome, species and genus identifiers are remapped.
- Make cross-genus parent-selection weighting configurable instead of a fixed multiplier.

## Reinforcement learning
- Add save/load support for the RL trainers, including the state needed to resume training.
- Implement a CUDA RL rollout backend, or document the CPU fallback.
- Review whether RL score normalisation should separate genomes by species rather than genus.

## Modules
- Bring `ConvSelfAttention` and `ConvCrossAttention` to feature parity for differential/branch modes.
- Complete `ConvCrossAttention` and test its supported input layouts.
- Dimensional stride and dilation for convolution and pooling modules.
- Running-statistics variants of batch normalisation and adaptive average pooling.

## Packaging
- GPU setup for AMD/Intel devices (CUDA pins are NVIDIA only).
