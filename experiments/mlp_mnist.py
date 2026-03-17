import os

from pandas._config import config
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.mlp import MLP
from metrics.geometry_metrics import tracking_step
from metrics.task1_gradient_buffer import Task1GradientBuffer
from exp_logger.csv_logger import CSVLogger

import torch
import torch.nn as nn
import torch.optim as optim
import os
from exp_logger.grad_logger import GradientLogger

def train_task1_to_convergence(model, loader, lr, device, checkpoint_dir, 
                               tolerance=1e-4, patience=3, max_epochs=500):
    """
    Stabilizes a standard model (e.g., MLP, ResNet) into a local minimum for Task 1.
    Halts dynamically when the epoch-level CrossEntropy loss stops improving.
    This ensures ||∇L1|| is driven to the noise floor prior to Geometric Vacuum tracking.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    

    print(f"Phase 1: Deepening Task 1 Basin (Auto-Converging up to {max_epochs} epochs)...")
    
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(max_epochs):
        running_loss = 0.0
        
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            
            
            # THE THEORY FLEX: Track the physical steepness of the manifold
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            
            optimizer.step()
            running_loss += loss.item()

        # Calculate average loss for the entire epoch
        epoch_loss = running_loss / len(loader)
        loss_change = best_loss - epoch_loss
        
        print(f"Task 1 Epoch {epoch+1:02d} | Loss: {epoch_loss:.4f} | Δ Loss: {loss_change:.6f} | ||∇L1||: {grad_norm:.4f}")
        
        # --- CONVERGENCE CHECK ---
        if abs(loss_change) < tolerance:
            patience_counter += 1
            print(f"  -> Flatline detected. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\n[CONVERGENCE REACHED] Model stabilized at Epoch {epoch+1}.")
                break
        else:
            # Reset patience if we get a meaningful, positive drop in loss
            if loss_change > 0:
                best_loss = epoch_loss
            patience_counter = 0

    # --- SAVE TASK 1 CHECKPOINT ---
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, "standard_model_task1.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch + 1,
        'final_loss': epoch_loss
    }, save_path)
    print(f"Task 1 Checkpoint saved to: {save_path}")
    
    return model

# -----------------------------
# Train Task 2 with measurement
# -----------------------------
def train_task2(
        model,
        task1_loader,
        task2_loader,
        results_path,
        epochs,
        lr,
        log_interval,
        buffer_batches,
        max_steps,
        device
):
    # Standard SGD is mandatory for exact ODE tracking
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    logger = CSVLogger(results_path)
    
    # 1. FIX: Define the fixed probe batch correctly
    fixed_iter = iter(task1_loader)
    fixed_x1, fixed_y1 = next(fixed_iter)
    fixed_x1, fixed_y1 = fixed_x1.to(device), fixed_y1.to(device)

    # Initialize integration accumulators
    logger.cum_pred_first = 0.0
    logger.cum_pred_second = 0.0
    logger.cum_pred_full = 0.0
    logger.cum_actual = 0.0
    logger.cum_stat_shift = 0.0 # Keep this for 1.0000 correlation parity

    step = 0

    # Initial baseline measurement
    model.eval()
    with torch.no_grad():
        prev_L1 = criterion(model(fixed_x1), fixed_y1).item()

    print(f"\nPhase 2: Tracking MLP Geometric ODE Forgetting...")
    step = 0
    grad_logger = GradientLogger(
        save_dir="grad_logs/experiment/mlp",
        interval=5   # save every 20 steps
    )
    for epoch in range(epochs):
        if step>max_steps: break
        for x2, y2 in tqdm(task2_loader, desc=f"Epoch {epoch}"):
            x2, y2 = x2.to(device), y2.to(device)
            step += 1
            # ----- 1. ODE Kinematic Measurement -----
            # FIX: Using 'fixed_y1' instead of the undefined 'y1'
            metrics, actual_drift, stat_shift, curr_L1_eval = tracking_step(
                model=model,
                optimizer=optimizer,
                fixed_x1=fixed_x1, 
                fixed_y1=fixed_y1, 
                x2=x2, 
                y2=y2, 
                lr=lr,
                prev_L1_eval=prev_L1,
                step = step,
                grad_logger = grad_logger
            )

            # Accumulate analytical predictions
            logger.cum_pred_first += metrics["first_order"]
            logger.cum_pred_second += metrics["second_order"]
            logger.cum_pred_full += metrics["full_law"]

            
            delta_actual = curr_L1_eval - prev_L1
            
            # If your model uses BatchNorm, stat_shift would be measured here.
            # For a standard MLP, it's usually 0, but we track it for unified parity.
            stat_shift = 0.0 
            
            logger.cum_actual += delta_actual
            logger.cum_stat_shift += stat_shift
            prev_L1 = curr_L1_eval

            # ----- 4. Logging -----
            logger.log({
                "step": step,
                "A": metrics["A"],
                "C": metrics["C"],
                "X": metrics["X"],
                "kappa": metrics["kappa"],
                "norm_g1": metrics["norm_g1"],
                "norm_g2": metrics["norm_g2"],
                "first_order_pred": metrics["first_order"],
                "second_order_pred": metrics["second_order"],
                "full_law_pred": metrics["full_law"],
                "cum_pred_first": logger.cum_pred_first,
                "cum_pred_second": logger.cum_pred_second,
                "cum_pred_full": logger.cum_pred_full,
                "cum_stat_shift": logger.cum_stat_shift,
                "total_forget": logger.cum_pred_full + logger.cum_stat_shift,
                "delta_L1_actual": delta_actual,
                "cum_actual": logger.cum_actual
            })
            

    logger.close()
    print(f"MLP Experiment complete. Results: {results_path}")

# -----------------------------
# Full Experiment
# -----------------------------

def run_experiment_mlp(dataset, results_path, config):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    task1_loader, task2_loader = dataset.get_task_loaders(
        config["batch_size"]
    )

    model = MLP().to(device)

    print("\nTraining Task 1\n")

    train_task1_to_convergence(
        model,
        task1_loader,
        lr=config["learning_rate"],
        checkpoint_dir='checkpoints',
        device=device,
        max_epochs=config["task1_epochs"],
    )
    # checkpoint_dir='checkpoints'
    # path = os.path.join(checkpoint_dir, "standard_model_task1.pth")
    # checkpoint = torch.load(path, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])

    print("\nTraining Task 2 with forgetting measurement\n")

    train_task2(
        model=model,
        task1_loader=task1_loader,
        task2_loader=task2_loader,
        results_path=results_path,
        epochs=config["task2_epochs"],
        lr=config["learning_rate"],
        log_interval=config["log_interval"],
        buffer_batches=config["buffer_batches"],
        max_steps=config['max_steps'],
        device=device
    )