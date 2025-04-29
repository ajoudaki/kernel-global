# Kernel Convergence in Deep Neural Networks

This repository contains code and materials for research on kernel convergence properties in deep neural networks, with a focus on Hermite polynomial analysis of activation functions.

## Repository Structure

- **`src/`** - Python source code
  - `utils.py` - Utilities for Hermite polynomial analysis and kernel calculations

- **`notebooks/`** - Jupyter notebooks
  - `evaluations.ipynb` - Main notebook with experiments and visualizations

- **`paper/`** - LaTeX files for the research paper
  - `main.tex` - Main paper document
  - `supplement.tex` - Supplementary material
  - `refs.bib` - Bibliography

- **`figures/`** - Generated figures for the paper
  - Various PDF files showing kernel convergence and ODE analysis

## Theory Background

### Hermite Polynomials

We use the recursive Hermite polynomial definition:
$$
He_0(x)=1, \qquad He_1(x)=x, \qquad He_k(x) = x He_{k-1}(x) - (k-1) He_{k-2}(x)
$$

And normalized Hermite polynomials:
$$
he_k(x) = \frac{1}{\sqrt{k!}}He_k(x) 
$$

**Orthogonality of Hermite polynomials:**
$$
E_{X\sim N(0,1)} [He_k(X) He_l(x)]= k! \delta_{kl},\qquad 
E_{X\sim N(0,1)} [he_k(X) he_l(x)]= \delta_{kl}
$$

*Expansion in Hermite basis:* Based on orthogonality, we can expand in the normalized basis as:
$$
f(x) = \sum_{k=0}^\infty c_k he_k(x), \qquad c_k = E_{X\sim N(0,1)}f(X)he_k(X)
$$

And in the non-normalized basis as:
$$
f(x) = \sum_{k=0}^\infty c_k He_k(x), \qquad c_k = \frac{1}{k!} E_{X\sim N(0,1)}f(X)He_k(X) 
$$

### Kernel Map

If $X,Y$ are standard Gaussian with covariance $\rho$, we define the kernel map $f$ as:
$$
\kappa_f(\rho) := E_{X,Y} f(X)f(Y) \qquad X,Y\sim N(0,1), E XY=\rho
$$

In the normalized Hermite basis we have:
$$
\kappa_f(\rho)=\sum_{k=0}^\infty c_k^2 \rho^k, \qquad f(x)=\sum_{k=0}^\infty c_k he_k(x)
$$

Basic properties of kernel map:
$$
E f(X) = c_0 = \sqrt{\kappa(0)}, \qquad E f(X)^2 = \sum_{k=0}^\infty c_k^2 = \kappa(1)
$$

## Features

The code in this repository includes:

1. Implementation of Hermite polynomial expansions for various activation functions
2. Analysis of kernel maps and their convergence properties
3. Evaluation of theoretical convergence bounds
4. Visualization of activation function dynamics
5. Notebook with complete experiments and paper figures

## Requirements

- Python 3.6+
- NumPy
- SciPy
- Matplotlib
- Jupyter Notebook
- PyTorch
- tqdm
- numba

## Paper Figures and Results

The paper figures and primary results are available in the `notebooks/evaluations.ipynb` notebook. The generated figures are stored in the `figures/` directory and include:

- `kernel_convergence.pdf` - Convergence analysis for tanh, relu, exp, and gelu
- `kernel_convergence_2.pdf` - Convergence analysis for selu, elu, celu, and sigmoid
- `kernel_convergence_leaky_relu.pdf` - Convergence analysis for leaky ReLU with various alpha values
- `kernel_ODE.pdf` - ODE analysis for various activation functions
- `kernel_ODE_fpi.pdf` - Fixed point iteration visualization

## License

[MIT License](LICENSE)