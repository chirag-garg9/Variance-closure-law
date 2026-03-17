import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM
from tqdm import tqdm

# Import the unified geometry engine you just verified
from metrics.geometry_metrics import tracking_step
from exp_logger.csv_logger import CSVLogger
from exp_logger.grad_logger import GradientLogger

# -----------------------------
# Train Task 1 to convergence (Warmup)
# -----------------------------
import torch
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm

def train_task1_to_convergence(model, loader, lr, device, checkpoint_dir, 
                               tolerance=5e-4, patience=10, window_size=100, max_steps=10000):
    """
    Stabilizes the model into a local minimum for WikiText (Task 1).
    Halts dynamically when the smoothed loss stops improving.
    This ensures ||∇L1|| is small enough to isolate the geometric X and C terms.
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()

    print("Phase 1: Deepening Task 1 Basin until Convergence...")
    step_count = 0
    loss_history = []
    best_sma = float('inf')
    patience_counter = 0
    
    # Use an iterator to seamlessly loop over the dataloader across epochs if needed
    data_iter = iter(loader)
    
    with tqdm(total=max_steps, desc="Task 1 Training") as pbar:
        while step_count < max_steps:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)
            
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x, labels=y)
            loss = outputs.loss
            loss.backward()
            
            # THE THEORY FLEX: Measure the exact gradient norm before stepping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            
            optimizer.step()
            
            loss_history.append(loss.item())
            step_count += 1
            pbar.update(1)
            
            # --- CONVERGENCE CHECK ---
            if step_count % window_size == 0:
                current_sma = np.mean(loss_history[-window_size:])
                loss_change = best_sma - current_sma
                
                tqdm.write(f"Step {step_count} | SMA Loss: {current_sma:.4f} | Δ Loss: {loss_change:.6f} | ||∇L1||: {grad_norm:.4f}")
                
                # If the loss improvement is smaller than our tolerance (or it slightly worsens)
                if abs(loss_change) < tolerance:
                    patience_counter += 1
                    tqdm.write(f"  -> Flatline detected. Patience: {patience_counter}/{patience}")
                    if patience_counter >= patience:
                        tqdm.write(f"\n[CONVERGENCE REACHED] Task 1 stabilized at step {step_count}.")
                        break
                else:
                    # Reset patience if we get a meaningful drop in loss
                    if loss_change > 0:
                        best_sma = current_sma
                    patience_counter = 0 

    # --- SAVE TASK 1 CHECKPOINT ---
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, "gpt2_task1_basin.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step_count,
        'final_loss': current_sma
    }, save_path)
    print(f"Task 1 Checkpoint saved to: {save_path}")
    
    return model

# -----------------------------
# Track Task 2 ODE Destruction
# -----------------------------
def track_task2_ode(model, t1_loader, t2_loader, results_path, steps, lr, log_interval, device):
    # CRITICAL: Strict SGD to match Taylor Expansion physics
    optimizer = optim.SGD(model.parameters(), lr=lr)
    logger = CSVLogger(results_path)
    
    # 1. Isolate the Fixed Probe Batch (The local Task 1 landscape)
    fixed_x1, fixed_y1 = next(iter(t1_loader))
    fixed_x1, fixed_y1 = fixed_x1.to(device), fixed_y1.to(device)
    
    # 2. Baseline Measurement
    model.eval()
    with torch.no_grad():
        out = model(fixed_x1)
        # Safely extract logits from the HF ModelOutput object
        logits = out.logits if hasattr(out, "logits") else (out[0] if isinstance(out, tuple) else out)
        
        prev_L1_eval = nn.functional.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)), 
            fixed_y1[:, 1:].contiguous().view(-1)
        ).item()
    grad_logger = GradientLogger(
        save_dir="grad_logs/experiment/gpt2",
        interval=5   # save every 20 steps
    )
    print(f"\nPhase 2: Tracking GPT-2 Geometric ODE Destruction (IMDB)...")
    step_count = 0
    
    for x2, y2 in tqdm(t2_loader, desc="ODE Track"):
        if step_count >= steps:
            break
            
        x2, y2 = x2.to(device), y2.to(device)

        # --- THE UNIFIED TRACKING STEP ---
        # We delegate the entire physics operation to your unified module
        metrics, actual_drift, stat_shift, curr_L1_eval = tracking_step(
            model=model, 
            optimizer=optimizer, 
            fixed_x1=fixed_x1, 
            fixed_y1=fixed_y1, 
            x2=x2, 
            y2=y2, 
            lr=lr, 
            prev_L1_eval=prev_L1_eval,
            step = step_count,
            grad_logger= grad_logger
        )

        # Update base for next step
        prev_L1_eval = curr_L1_eval

        # Accumulate logger metrics
        logger.cum_pred_first += metrics["first_order"]
        logger.cum_pred_second += metrics["second_order"]
        logger.cum_pred_full += metrics["full_law"]
        logger.cum_actual += actual_drift

        # --- LOGGING ---
        if step_count % log_interval == 0:
            logger.log({
                "step": step_count,
                "A": metrics["A"],
                "C": metrics["C"],
                "X": metrics["X"],
                "kappa": metrics["kappa"],
                "norm_g1": metrics["norm_g1"],
                "norm_g2": metrics["norm_g2"],
                "stat_shift": stat_shift,
                "first_order_pred": metrics["first_order"],
                "second_order_pred": metrics["second_order"],
                "full_law_pred": metrics["full_law"],
                "cum_pred_first": logger.cum_pred_first,
                "cum_pred_second": logger.cum_pred_second,
                "cum_pred_full": logger.cum_pred_full,
                "total_forget": logger.cum_pred_full + stat_shift,
                "delta_L1_actual": actual_drift,
                "cum_actual": logger.cum_actual
            })
        
        step_count += 1

    logger.close()
    print(f"\nExperiment complete. Data saved to: {results_path}")

# -----------------------------
# Main Entry Setup
# -----------------------------
def run_llm_experiment(dataset,results_path, config):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Task loaders from your dataset setup
    t1_loader, t2_loader = dataset.get_task_loaders(config["batch_size"])
    
    # Load proper HF Model
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"], 
        attn_implementation="eager"  # Force standard math attention
    ).to(device)

    # 1. Stabilize the WikiText Basin
    train_task1_to_convergence(
        model, 
        t1_loader,  
        config["task1_lr"], 
        device,
        checkpoint_dir=config["checkpoint_dir"],
        max_steps=config['task1_steps']
    )

    # save_path = os.path.join(config["checkpoint_dir"], "gpt2_task1_basin.pth")
    # checkpoint = torch.load(save_path,weights_only=False)

    # model.load_state_dict(checkpoint["model_state_dict"])
    
    # 2. Track continuous forgetting on IMDB
    track_task2_ode(
        model, 
        t1_loader, 
        t2_loader, 
        config["results_path"], 
        config["task2_steps"], 
        config["task2_lr"], 
        config["log_interval"], 
        device
    )