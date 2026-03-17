import os
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Suppress technical convergence warnings
warnings.simplefilter('ignore', ConvergenceWarning)

def parse_args():
    parser = argparse.ArgumentParser(description="Law of Forgetting: Complete Research Pipeline")
    
    # Path configuration
    parser.add_argument("--data_dir", type=str, required=True, help="Root folder containing architecture subdirectories")
    parser.add_argument("--main_out", type=str, default="./paper_assets/main_text", help="Output for main figures")
    parser.add_argument("--appx_out", type=str, default="./paper_assets/appendix", help="Output for technical appendices")
    
    # Model configuration
    parser.add_argument("--archs", type=str, nargs="+", 
                        default=["mlp_mnist", "resnet_cifar10", "vit_cifar100", 
                                 "gpt2_forgetting_ode_10k", "mamba_forgetting_ode", "gnn_forgetting_ode_10k"])
    parser.add_argument("--rep_model", type=str, default="mlp_mnist", help="Model used for Figure 1 deep-dive")
    
    # Column configuration
    parser.add_argument("--col_delta", default="delta_L1_actual")
    parser.add_argument("--col_A", default="A")
    parser.add_argument("--col_g1", default="norm_g1")
    parser.add_argument("--col_g2", default="norm_g2")
    parser.add_argument("--col_pred", default="first_order_pred")
    
    return parser.parse_args()

def get_longest_contiguous_block(c_array):
    valid_mask = ~np.isnan(c_array)
    if not np.any(valid_mask): return np.array([])
    blocks = np.split(c_array[valid_mask], np.where(np.diff(np.where(valid_mask)[0]) != 1)[0] + 1)
    return max(blocks, key=len)

def main():
    args = parse_args()
    
    # Styling
    plt.rcParams.update({
        'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14,
        'legend.fontsize': 11, 'lines.linewidth': 1.5, 'figure.dpi': 300, 'font.family': 'serif'
    })
    os.makedirs(args.main_out, exist_ok=True)
    os.makedirs(args.appx_out, exist_ok=True)

    # Global Data Storage
    results_table = []
    global_stats = {"obs": [], "pred": [], "deff": [], "labels": [], "id_x": [], "id_y": []}
    seed_robustness = {}
    fig1_data = {}

    print(f"--- 1. Processing Architectures from {args.data_dir} ---")

    for arch in args.archs:
        display_name = arch.split('_')[0].upper()
        files = glob.glob(os.path.join(args.data_dir, arch, "*seed*.csv"))
        if not files: 
            files = glob.glob(os.path.join(args.data_dir, f"{arch}*seed*.csv"))
        if not files: continue

        arch_vars_obs, arch_vars_pred = [], []
        arch_x, arch_y = [], []

        for f in files:
            df = pd.read_csv(f)
            delta = df[args.col_delta].values
            A, g1, g2 = df[args.col_A].values, df[args.col_g1].values, df[args.col_g2].values
            
            lr = df["learning_rate"].values if "learning_rate" in df.columns else \
                 np.full(len(df), np.abs(df[args.col_pred] / (A + 1e-9)).median())
            
            c = A / (g1 * g2 + 1e-9)
            c_t, c_tp1 = c[:-1], c[1:]
            valid = (~np.isnan(c_t)) & (~np.isnan(c_tp1))
            c_t, c_tp1 = c_t[valid], c_tp1[valid]

            if len(c_t) > 10:
                # Calculate mean-reversion coefficient k
                k = -np.polyfit(c_t, c_tp1 - c_t, 1)[0]
                eps = c_tp1 - (1 - k) * c_t
                var_c = np.var(c[~np.isnan(c)])
                sigma2 = np.var(eps)

                arch_vars_obs.append(var_c)
                arch_vars_pred.append(sigma2 / (2*k - k**2))
                arch_x.append(-lr * A)
                arch_y.append(delta)

                # Capture representative data for Fig 1
                if args.rep_model in arch and "c_t" not in fig1_data:
                    fig1_data = {"ct": c_t, "dct": c_tp1 - c_t, "eps": eps, "k": k}

        # Aggregate Statistics
        m_obs, m_pred = np.mean(arch_vars_obs), np.mean(arch_vars_pred)
        global_stats["obs"].append(m_obs)
        global_stats["pred"].append(m_pred)
        global_stats["deff"].append(1/m_obs)
        global_stats["labels"].append(display_name)
        global_stats["id_x"].append(np.concatenate(arch_x))
        global_stats["id_y"].append(np.concatenate(arch_y))
        
        seed_robustness[display_name] = {
            "obs_m": m_obs, "obs_s": np.std(arch_vars_obs),
            "pred_m": m_pred, "pred_s": np.std(arch_vars_pred)
        }

        results_table.append({
            "Model": display_name, "k": f"{k:.4f}", "Var_Obs": f"{m_obs:.5f}", 
            "d_eff": f"{1/m_obs:.1f}", "Pred_Error": f"{abs(m_obs-m_pred)/m_obs*100:.2f}%"
        })

    # --- SAVE TABLE 1 ---
    pd.DataFrame(results_table).to_csv(os.path.join(args.main_out, "Table_1_Main.csv"), index=False)

    # --- FIGURE 1: DYNAMICS ---
    print("Generating Figure 1...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    # Panel 1: Drift
    bins = np.linspace(np.percentile(fig1_data["ct"], 1), np.percentile(fig1_data["ct"], 99), 20)
    b_mean, b_edge, _ = binned_statistic(fig1_data["ct"], fig1_data["dct"], bins=bins)
    axes[0].scatter(fig1_data["ct"], fig1_data["dct"], alpha=0.05, color='gray', s=5)
    axes[0].plot((b_edge[:-1]+b_edge[1:])/2, b_mean, 'ro-', label="Empirical Drift")
    axes[0].plot(b_edge, -fig1_data["k"]*b_edge, 'k--', label=f"-k*c (k={fig1_data['k']:.3f})")
    axes[0].set_title("Mean-Reverting Drift"); axes[0].legend()
    # Panel 2: Trajectory
    axes[1].plot(fig1_data["ct"][:100], color='black', label="Actual")
    axes[1].plot((1-fig1_data["k"])*fig1_data["ct"][:99], 'r--', label="Predicted")
    axes[1].set_title("Cosine Trajectory"); axes[1].legend()
    # Panel 3: Noise
    axes[2].hist(fig1_data["eps"], bins=50, color='teal', alpha=0.7, density=True)
    axes[2].set_title("Residual Noise Distribution")
    plt.tight_layout(); fig.savefig(os.path.join(args.main_out, "Fig_1_Dynamics.pdf"))

    # --- FIGURE 2: GLOBAL LAWS ---
    print("Generating Figure 2...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    # Panel 1: Var Prediction
    lims = [0, max(global_stats["obs"])*1.1]
    axes[0].plot(lims, lims, 'k--', alpha=0.5)
    for i in range(len(global_stats["labels"])):
        axes[0].scatter(global_stats["obs"][i], global_stats["pred"][i], label=global_stats["labels"][i], s=100)
    axes[0].set_title("Second-Order Law Accuracy"); axes[0].legend(fontsize=8)
    # Panel 2: Deff
    axes[1].bar(global_stats["labels"], global_stats["deff"], color='royalblue')
    axes[1].set_title("Effective Dimension $d_{eff}$"); plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
    # Panel 3: Identity
    ix, iy = np.concatenate(global_stats["id_x"]), np.concatenate(global_stats["id_y"])
    axes[2].scatter(ix[::10], iy[::10], alpha=0.1, color='purple', s=1)
    axes[2].plot([min(ix), max(ix)], [min(ix), max(ix)], 'k--')
    axes[2].set_title("First-Order Forgetting Identity")
    plt.tight_layout(); fig.savefig(os.path.join(args.main_out, "Fig_2_Global_Laws.pdf"))

    # --- APPENDIX D: RESNET DEEP DIVE ---
    print("Running ResNet Deep-Dive...")
    res_f = glob.glob(os.path.join(args.data_dir, "*resnet*", "*seed*.csv"))
    if res_f:
        df_r = pd.read_csv(res_f[0])
        c_raw = df_r[args.col_A].values / (df_r[args.col_g1].values * df_r[args.col_g2].values + 1e-9)
        c_clean = pd.Series(get_longest_contiguous_block(c_raw))
        detrended = (c_clean - c_clean.rolling(500, center=True).mean()).dropna()
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(c_clean, color='gray', alpha=0.3); ax[0].plot(c_clean.rolling(500).mean(), color='red')
        ax[0].set_title("ResNet Global Drift")
        ax[1].plot(detrended, color='blue', alpha=0.6)
        ax[1].set_title(f"Detrended (ADF p={adfuller(detrended.values)[1]:.1e})")
        plt.tight_layout(); fig.savefig(os.path.join(args.appx_out, "Appx_ResNet_Stationarity.pdf"))

    print(f"\nPipeline Complete. Assets in {args.main_out} and {args.appx_out}")

if __name__ == "__main__":
    main()