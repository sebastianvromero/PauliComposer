# PauliComposer
We introduce a simple algorithm that efficiently computes tensor products of Pauli matrices. This is done by tailoring the calculations to this specific case, which allows to avoid unnecessary calculations. The strength of this strategy is benchmarked against state-of-the-art techniques, showing a remarkable acceleration. As a side product, we provide an optimized method for one key calculus in quantum simulations: the Pauli basis decomposition of Hamiltonians.

Pre-print at [arXiv:2301.00560](https://arxiv.org/abs/2301.00560).

## Cite us

If you use PauliComposer in your work, thanks for your interest and please cite our corresponding manuscript as:
```
@misc{paulicomposer,
    title={PauliComposer: Compute Tensor Products of Pauli Matrices Efficiently}, 
    author={Sebastián V. Romero, and Juan Santos-Suárez},
    year={2023},
    eprint={2301.00560},
    archivePrefix={arXiv},
    primaryClass={quant-ph}
}
```