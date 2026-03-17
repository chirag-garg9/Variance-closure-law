import yaml
import argparse
import os
import torch
import numpy as np
import random

from experiments import *
from datasets_our import *

EXPERIMENT_REGISTRY = {
    'mlp': run_experiment_mlp,
    'resnet_cifar': run_resnet_cifar_experiment,
    'vit_cifar': run_vit_experiment,
    'llm': run_llm_experiment,
    'mamba_llm' : run_mamba_experiment,
    'gnn_arxiv_pretrained' : run_pretrained_gnn_experiment,
    'resnet_cifar_fixed': run_resnet_cifar_experiment_fixed,
}

DATASET_REGISTRY = {
    "mnist_split": MNISTSplit,
    "cifar10_split": CIFAR10Split,
    "cifar100_split": CIFAR100Split,
    "wiki_imdb": LLMDomainSplit,
    "ogbn-arxiv": ArxivDomainSplit
}

def set_seed(seed):
    """Ensure absolute reproducibility for each independent run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Essential for 1.0000 correlation consistency
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_dataset(name):
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset {name}")
    return DATASET_REGISTRY[name]()

def build_experiment(name):
    if name not in EXPERIMENT_REGISTRY:
        raise ValueError(f"Unknown experiment {name}")
    return EXPERIMENT_REGISTRY[name]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seeds", type=int, default=5, help="Number of independent runs")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Define your seed bank
    seed_list = [7] # 5 distinct seeds
    num_runs = min(args.seeds, len(seed_list))

    base_results_path = config["results_path"]

    for i in range(num_runs):
        current_seed = seed_list[i]
        print(f"\n" + "="*50)
        print(f"RUNNING SEED {current_seed} ({i+1}/{num_runs})")
        print("="*50 + "\n")

        # 1. Set the global seed
        set_seed(current_seed)

        # 2. Update results path so we don't overwrite previous runs
        # e.g., results/mamba_forgetting.csv -> results/mamba_forgetting_seed42.csv
        name, ext = os.path.splitext(base_results_path)
        config["results_path"] = f"{name}_seed{current_seed}{ext}"
        config["seed"] = current_seed
        # 3. Build fresh dataset and model for each seed
        # This ensures we start from a different random initialization
        dataset = build_dataset(config["dataset"])
        experiment = build_experiment(config['experiment'])
        
        # 4. Execute the experiment
        experiment(
            dataset=dataset,
            results_path=config["results_path"],
            config=config
        )

    print(f"\nAll {num_runs} seeds complete. Ready for statistical analysis.")

if __name__ == "__main__":
    main()