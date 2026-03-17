import os
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Suppress convergence warnings for cleaner CLI output
warnings.simplefilter('ignore', ConvergenceWarning)

def parse_args():
    parser = argparse.ArgumentParser(description="Monster Appendix Suite Generator")
    
    # Path configuration
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Path to the results directory.")
    parser.add_argument("--output_dir", type=str, default="./paper_assets/monster_appendix",
                        help="Where to save the PDF plots.")
    
    # Architecture and column mapping
    parser.add_argument("--archs", type=str, nargs="+", 
                        default=["mlp_mnist", "resnet_cifar10", "vit_cifar100", 
                                 "gpt2_forgetting_ode_10k", "mamba_forgetting_ode", "gnn_forgetting_ode_10k"],
                        help="List of architecture folder names.")
    parser.add_argument("--col_delta", default="delta_L1_actual")
    parser.add_argument("--col_A", default="A")
    parser.add_argument("--col_g1", default="norm_g1")
    parser.add_argument("--col_g2", default="norm_g2")
    parser.add_argument("--col_pred", default="first_order_pred")
    
    # Hyperparameters
    parser.add_argument("--window", type=int, default=500, help="Rolling window size.")
    parser.add_argument("--dpi", type=int, default=300)
    
    return parser.parse_args()

def get_longest_contiguous_block(c_array):
    valid_mask = ~np.isnan(c_array)
    if not np.any(valid_mask): return np.array([])
    blocks = np.split(c_array[valid_mask], np.where(np.diff(np.where(valid_mask)[0]) != 1)[0] + 1)
    return max(blocks, key=len)

def main():
    args = parse_args()
    
    # Global Plot Config
    plt.rcParams.update({
        'font.size': 10, 'axes.labelsize': 12, 'axes.titlesize': 12,
        'legend.fontsize': 10, 'lines.linewidth': 1.5, 'figure.dpi': args.dpi,
        'font.family': 'serif'
    })
    os.makedirs(args.output_dir, exist_ok=True)

    # Data containers
    storage = {
        "identity_errors": {},
        "timescale_data": {},
        "acf_pacf_data": {},
        "rolling_stats": {}
    }

    print(f"--- Processing Data from {args.data_dir} ---")

    for arch_name in args.archs:
        display_name = arch_name.split('_')[0].upper()
        
        # Robust file finding
        files = glob.glob(os.path.join(args.data_dir, arch_name, "*seed*.csv"))
        if not files: 
            files = glob.glob(os.path.join(args.data_dir, f"{arch_name}*seed*.csv"))
        if not files:
            print(f"Skipping {arch_name}: No CSV files found.")
            continue

        df = pd.read_csv(files[0])
        
        # Column Extraction
        delta = df[args.col_delta].values
        A, g1, g2 = df[args.col_A].values, df[args.col_g1].values, df[args.col_g2].values
        
        # Calculate LR if missing
        if "learning_rate" in df.columns:
            lr = df["learning_rate"].values
        else:
            lr = np.full(len(df), np.abs(df[args.col_pred] / (A + 1e-9)).median())
        
        c_raw = A / (g1 * g2 + 1e-9)
        c_contig = get_longest_contiguous_block(c_raw)
        
        # A. Identity Error
        pred_delta = -lr * A
        storage["identity_errors"][display_name] = (delta - pred_delta)[~np.isnan(delta) & ~np.isnan(pred_delta)]
        
        # B. Timescale Separation (First 500 samples after transient)
        start = len(c_raw)//10
        g1_w = g1[start:start+args.window]
        c_w = c_raw[start:start+args.window]
        scale = lambda x: (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-9)
        storage["timescale_data"][display_name] = {"g1": scale(g1_w), "c": scale(c_w)}
        
        # C. PACF
        c_t, c_tp1 = c_contig[:-1], c_contig[1:]
        k_est = -np.polyfit(c_t, c_tp1 - c_t, 1)[0]
        eps = c_tp1 - (1 - k_est) * c_t
        storage["acf_pacf_data"][display_name] = {"pacf": pacf(eps, nlags=15)}
        
        # D. Rolling Stats
        c_ser = pd.Series(c_contig)
        storage["rolling_stats"][display_name] = {
            "mean": c_ser.rolling(window=args.window, center=True).mean().values,
            "var": c_ser.rolling(window=args.window, center=True).var().values,
            "raw": c_contig
        }

    # Helper: Grid plotting to avoid repetitive code
    def plot_appendix(title, filename, data_dict, plot_func):
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        fig.suptitle(title, fontsize=16)
        for ax, (model, data) in zip(axes.flatten(), data_dict.items()):
            plot_func(ax, model, data)
        plt.tight_layout()
        fig.savefig(os.path.join(args.output_dir, filename))
        plt.close()

    # --- Generate Appx A: Residuals ---
    def plot_a(ax, model, errs):
        ax.hist(errs, bins=100, color='indigo', alpha=0.7, density=True)
        ax.axvline(0, color='black', linestyle='--')
        ax.set_title(model); ax.set_yscale('log')
    plot_appendix("Appendix A: Taylor Expansion Residuals", "Appx_A_Residuals.pdf", storage["identity_errors"], plot_a)

    # --- Generate Appx B: Timescales ---
    def plot_b(ax, model, data):
        ax.plot(data["g1"], color='darkred', label="Grad Norm")
        ax.plot(data["c"], color='steelblue', alpha=0.6, label="Cosine")
        ax.set_title(model)
        if model == list(storage["timescale_data"].keys())[0]: ax.legend()
    plot_appendix("Appendix B: Timescale Separation", "Appx_B_Timescales.pdf", storage["timescale_data"], plot_b)

    # --- Generate Appx C: PACF ---
    def plot_c(ax, model, data):
        ax.bar(range(len(data["pacf"])), data["pacf"], color='gray', edgecolor='black')
        ax.axhline(0, color='black', linewidth=1); ax.set_ylim(-0.2, 0.5)
        ax.set_title(f"{model} PACF")
    plot_appendix("Appendix C: Residual Memory Fingerprints", "Appx_C_PACF.pdf", storage["acf_pacf_data"], plot_c)

    # --- Generate Appx D: Rolling Stationarity ---
    def plot_d(ax, model, data):
        ax.plot(data["raw"], color='lightgray', alpha=0.3)
        ax.plot(data["mean"], color='blue', label="Mean")
        ax2 = ax.twinx()
        ax2.plot(data["var"], color='red', label="Var")
        ax.set_title(model)
    plot_appendix("Appendix D: Global Non-Stationarity", "Appx_D_Rolling.pdf", storage["rolling_stats"], plot_d)

    # --- Generate Appx E: Deff ---
    def plot_e(ax, model, data):
        deff = 1.0 / (data["var"] + 1e-9)
        ax.plot(deff, color='forestgreen')
        ax.set_title(fr"{model} (Median $d_{{eff}} \approx {np.nanmedian(deff):.0f}$)")
        ax.set_yscale('log')
    plot_appendix("Appendix E: Evolution of Effective Dimension", "Appx_E_Deff.pdf", storage["rolling_stats"], plot_e)

    # --- Appx F: ResNet ARMA Heatmap (Deep Dive) ---
    if "RESNET" in storage["rolling_stats"]:
        print("Generating Appx F: ResNet ARMA Heatmap...")
        rc = storage["rolling_stats"]["RESNET"]["raw"]
        c_detrend = (pd.Series(rc) - pd.Series(rc).rolling(args.window, center=True).mean()).dropna().values
        p_max, q_max = 4, 4
        m = np.zeros((p_max, q_max))
        for p in range(1, p_max+1):
            for q in range(1, q_max+1):
                try:
                    res = ARIMA(c_detrend, order=(p, 0, q)).fit()
                    m[p-1, q-1] = acorr_ljungbox(res.resid, lags=[10], return_df=True)["lb_pvalue"].values[0]
                except: pass
        
        fig, ax = plt.subplots(figsize=(7, 6))
        cax = ax.imshow(m, cmap='RdYlGn', vmin=0, vmax=0.1, origin='lower')
        for i in range(p_max):
            for j in range(q_max):
                ax.text(j, i, f"{m[i,j]:.3f}", ha="center", va="center", fontweight='bold')
        fig.colorbar(cax, label="Ljung-Box p-value")
        ax.set_title("Appx F: ResNet ARMA Grid Search")
        fig.savefig(os.path.join(args.output_dir, "Appx_F_ResNet_ARMA.pdf"))

    print(f"Success! All assets saved to: {args.output_dir}")

if __name__ == "__main__":
    main()