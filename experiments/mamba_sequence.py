import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM
import numpy as np

# Use the EXACT SAME unified engine used for GPT-2 and GNNs
from metrics.geometry_metrics import tracking_step
from exp_logger.csv_logger import CSVLogger
from exp_logger.grad_logger import GradientLogger

def train_mamba_task1_to_convergence(model, loader, lr, device, checkpoint_dir, 
                                     tolerance=5e-4, patience=10, window_size=10000, max_steps=10000):
    """
    Stabilizes the Mamba readout head into a local minimum for Task 1.
    Halts dynamically when the smoothed loss stops improving.
    This ensures ||∇L1|| is driven to the noise floor on the projection manifold.
    """
    # Isolate the trainable head for the optimizer and gradient norm tracking
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    
    model.train()
    print("Phase 1: Deepening Mamba Task 1 Basin (Readout Head Only)...")
    
    step_count = 0
    loss_history = []
    best_sma = float('inf')
    patience_counter = 0
    current_sma = 0
    
    # Use an iterator to seamlessly loop over the dataloader across epochs
    data_iter = iter(loader)
    
    with tqdm(total=max_steps, desc="Task 1 Mamba Training") as pbar:
        while step_count < max_steps:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)
            
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=x, labels=y)
            loss = outputs.loss
            loss.backward()
            
            # THE THEORY FLEX: Measure the gradient norm ONLY for the active parameters
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float('inf'))
            
            optimizer.step()
            
            loss_history.append(loss.item())
            step_count += 1
            pbar.update(1)
            
            # --- CONVERGENCE CHECK ---
            if step_count % window_size == 0:
                current_sma = np.mean(loss_history[-window_size:])
                loss_change = best_sma - current_sma
                
                tqdm.write(f"Step {step_count} | SMA Loss: {current_sma:.4f} | Δ Loss: {loss_change:.6f} | ||∇L1_head||: {grad_norm:.4f}")
                
                # If the loss improvement is smaller than our tolerance
                if abs(loss_change) < tolerance:
                    patience_counter += 1
                    tqdm.write(f"  -> Flatline detected. Patience: {patience_counter}/{patience}")
                    if patience_counter >= patience:
                        tqdm.write(f"\n[CONVERGENCE REACHED] Mamba Readout Head stabilized at step {step_count}.")
                        break
                else:
                    # Reset patience if we get a meaningful drop in loss
                    if loss_change > 0:
                        best_sma = current_sma
                    patience_counter = 0 

    # --- SAVE TASK 1 CHECKPOINT ---
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, "mamba_head_task1.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step_count,
        'final_loss': best_sma
    }, save_path)
    print(f"Mamba Task 1 Checkpoint saved to: {save_path}")
    
    return model

def track_mamba_task2(model, t1_loader, t2_loader, results_path, steps, lr, log_interval, device):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(trainable_params, lr=lr)
    
    logger = CSVLogger(results_path)
    logger.cum_pred_first = 0.0
    logger.cum_pred_second = 0.0
    logger.cum_pred_full = 0.0
    logger.cum_actual = 0.0
    logger.cum_stat_shift = 0.0 
    
    fixed_x1, fixed_y1 = next(iter(t1_loader))
    fixed_x1, fixed_y1 = fixed_x1.to(device), fixed_y1.to(device)
    
    model.eval()
    with torch.no_grad():
        out = model(input_ids=fixed_x1)
        logits = out.logits if hasattr(out, "logits") else (out[0] if isinstance(out, tuple) else out)
        prev_L1_eval = nn.functional.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)).to(torch.float64), 
            fixed_y1[:, 1:].contiguous().view(-1)
        ).item()

    print(f"\nPhase 2: Exact Geometric ODE Tracking of Mamba Readout Head...")
    grad_logger = GradientLogger(
        save_dir="grad_logs/experiment/mamba",
        interval=5   # save every 20 steps
    )
    for step_count, (x2, y2) in enumerate(tqdm(t2_loader, desc="Mamba Head Track")):
        if step_count >= steps:
            break
            
        x2, y2 = x2.to(device), y2.to(device)

        # PLUG AND PLAY: Exact mathematical tracking using torch.func
        metrics, actual_drift, stat_shift, curr_L1_eval = tracking_step(
            model=model, optimizer=optimizer, fixed_x1=fixed_x1, fixed_y1=fixed_y1, 
            x2=x2, y2=y2, lr=lr, prev_L1_eval=prev_L1_eval,step=step_count,grad_logger=grad_logger
        )
        delta_actual = curr_L1_eval - prev_L1_eval
        logger.cum_actual += delta_actual
        prev_L1_eval = curr_L1_eval

        logger.cum_pred_first += metrics["first_order"]
        logger.cum_pred_second += metrics["second_order"]
        logger.cum_pred_full += metrics["full_law"]
        logger.cum_stat_shift += stat_shift

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
                    "delta_L1_actual": delta_actual,
                    "cum_actual": logger.cum_actual
                })

    logger.close()
    print(f"\nExperiment complete. Golden Standard Data saved to: {results_path}")

def run_mamba_experiment(dataset, results_path, config):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    t1_loader, t2_loader = dataset.get_task_loaders(config["batch_size"])
    
    print(f"Loading Mamba Model: {config['model_name']}...")
    model = AutoModelForCausalLM.from_pretrained(config["model_name"]).to(device)

    # ==========================================
    # THE FREEZE: Freeze everything except the Head
    # ==========================================
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. Break the Tied Embeddings (Crucial for Head-Only tracking)
    # This copies the weights into a new, independent parameter
    model.lm_head.weight = nn.Parameter(model.lm_head.weight.clone())
    
    # 3. Unfreeze ONLY the newly independent head
    for param in model.lm_head.parameters():
        param.requires_grad = True
            
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Backbone frozen. Tracking exact ODE geometry for {trainable_params} parameters in the LM Head.")

    train_mamba_task1_to_convergence(
        model, t1_loader, config["task1_lr"], device, config["checkpoint_dir"],max_steps=config["task1_steps"]
    )

    # save_path = os.path.join(config["checkpoint_dir"], "mamba_task1_basin.pth")
    # checkpoint = torch.load(save_path,weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    
    track_mamba_task2(
        model, t1_loader, t2_loader, results_path, config["task2_steps"], 
        config["task2_lr"], config["log_interval"], device
    )