# GLI: Laws of Gradient Interference

Official implementation of the paper: **"Gradient Interference Exhibits Low-Dimensional Stochastic Dynamics with Variance Closure"**

This repository provides the tools to measure, model, and predict task interference in deep neural networks. We show that gradient interference is not a high-dimensional chaos but a low-dimensional stochastic process governed by measurable macroscopic laws.

---

## 🚀 Quick Start

### 1. Environment Setup
First, create a virtual environment to manage dependencies:

```bash
# Create the environment
conda create -n gli python=3.10 -y
conda activate gli

# Install required packages
pip install -r requirements.txt
```

### 2. Running Experiments
The core optimization logic is driven by configuration files. To run a specific architecture (e.g., ResNet-20 on CIFAR-10) under the continual learning protocol

```bash
python main.py --config configs/resnet20_cifar10.yaml
```

Supported architectures in the `configs/` directory include:
* **MLP**: Dense architectures for Split MNIST
* **ResNet-20**: Convolutional networks with skip-connections for Split Cifar 10
* **ViT-Tiny**: Vision Transformers for CIFAR-100.
* **Mamba**: State-space modeling for sequences.
* **GPT-2 Small**: Transformer-based language modeling.
* **GNN**: Graph Convolutional Networks for temporal splits.

---

## 📊 Running Analysis

Once trajectories are generated, you can validate the macroscopic laws using the analysis suite.

### Variance Closure & Effective Dimension
To generate the **Variance Parity** plots and calculate the effective dimension $d_{eff} = \frac{1}{Var(c)}:

```bash
python analysis.py --task closure --log_dir logs/your_experiment_path
```

### Time-Series Diagnostics
To perform the **Stationarity** tests (ADF) and **Memory Structure** analysis (ACF/PACF/ARMA):

```bash
python analysis.py --task diagnostics --log_dir logs/your_experiment_path
```
