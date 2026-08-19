# DeePKS-kit

DeePKS-kit is a program to generate accurate energy functionals for quantum chemistry systems, for both perturbative scheme (DeePHF) and self-consistent scheme (DeePKS).

The program provides a command line interface `deepks` that contains five sub-commands, 
- `train`: train an neural network based post-HF energy functional model
- `test`: test the post-HF model with given data and show statistics
- `scf`: run self-consistent field calculation with given energy model
- `stats`: collect and print statistics of the SCF the results
- `iterate`: iteratively train an self-consistent model by combining four commands above

## Installation

### Using uv

The repository includes a locked development environment managed by [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --python 3.11
uv run deepks --help
uv run pytest
```

## Usage

An relatively detailed decrisption of the `deepks-kit` library can be found in [here](https://arxiv.org/pdf/2012.14615.pdf). Please also refer to the reference for the description of methods.

Please see [`examples`](./examples) folder for the usage of `deepks-kit` library. A detailed example with executable data for single water molecules can be found [here](./examples/water_single). A more complicated one for training water clusters can be found [here](./examples/water_cluster).

Check [this input file](./examples/water_cluster/args.yaml) for detailed explanation for possible input parameters, and also [this one](./examples/water_cluster/shell.yaml) if you would like to run on local machine instead of using Slurm scheduler.

## References

[1] Chen, Y., Zhang, L., Wang, H. and E, W., 2020. Ground State Energy Functional with Hartree–Fock Efficiency and Chemical Accuracy. The Journal of Physical Chemistry A, 124(35), pp.7155-7165.

[2] Chen, Y., Zhang, L., Wang, H. and E, W., 2021. DeePKS: A Comprehensive Data-Driven Approach toward Chemically Accurate Density Functional Theory. Journal of Chemical Theory and Computation, 17(1), pp.170–181.


<!-- ## Original TODO

- [ ] Print loss separately for E and F in training.
- [ ] Rewrite all `print` function using `logging`.
- [ ] Write a detailed README and more docs.
- [ ] Add unit tests. -->
