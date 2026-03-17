# Auto-generated file
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from models.vit import get_proper_vit_tiny
from datasets_our.cifar_split import CIFAR100Split
from metrics.geometry_metrics import tracking_step 
from metrics.task1_gradient_buffer import Task1GradientBuffer
from exp_logger.csv_logger import CSVLogger
from exp_logger.grad_logger import GradientLogger

# -----------------------------
# Train Task 1 to convergence
# -----------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import os

def train_task1_to_convergence(model, loader, lr, device, checkpoint_dir, max_epochs=200, tolerance=1e-4, patience=3):
    # AdamW is standard for fine-tuning pre-trained ViT backbones
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()
    model.train()

    print(f"Phase 1: Fine-tuning ViT-Small on Task 1 (Auto-Converging up to {max_epochs} epochs)...")
    
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(max_epochs):
        running_loss = 0.0
        
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            loss = criterion(model(x), y)
            loss.backward()
            
            # THE THEORY FLEX: ViT landscapes are sharp. Track the physical steepness.
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(loader)
        loss_change = best_loss - epoch_loss
        
        print(f"Task 1 Epoch {epoch+1:02d} | Loss: {epoch_loss:.4f} | Δ Loss: {loss_change:.6f} | ||∇L1||: {grad_norm:.4f}")

        # --- CONVERGENCE CHECK ---
        if abs(loss_change) < tolerance:
            patience_counter += 1
            print(f"  -> ViT basin flatline detected. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\n[CONVERGENCE REACHED] ViT manifold geometry stabilized at Epoch {epoch+1}.")
                break
        else:
            # Only reset patience if we get a meaningful, positive drop in loss
            if loss_change > 0:
                best_loss = epoch_loss
            patience_counter = 0

    # --- SAVE TASK 1 CHECKPOINT ---
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, "vit_task1_basin.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch + 1,
        'final_loss': epoch_loss
    }, save_path)
    print(f"Task 1 Checkpoint saved to: {save_path}")
    
    return model
# -----------------------------
# Train Task 2 with continuous measurement
# -----------------------------
def train_task2(model, task1_loader, task2_loader, results_path, epochs, lr, log_interval, buffer_batches,max_steps, device):
    # CRITICAL: SGD only for Task 2 to match Taylor Expansion physics
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    buffer = Task1GradientBuffer(model, task1_loader, buffer_batches, device)
    logger = CSVLogger(results_path)
    
    # 1. Isolate the Fixed Probe Batch
    fixed_x1, fixed_y1 = next(iter(task1_loader))
    fixed_x1, fixed_y1 = fixed_x1.to(device), fixed_y1.to(device)
    
    step = 0
    
    # 2. Baseline Measurement (Strictly in eval() for pure geometry)
    model.eval()
    with torch.no_grad():
        prev_L1_eval = criterion(model(fixed_x1), fixed_y1).item()

    print(f"\nPhase 2: Tracking ViT Geometric ODE Destruction...")
    step = 0
    grad_logger = GradientLogger(
        save_dir="grad_logs/experiment/vit",
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
                prev_L1_eval=prev_L1_eval,
                step = step,
                grad_logger = grad_logger
            )

            # Accumulate analytical predictions
            logger.cum_pred_first += metrics["first_order"]
            logger.cum_pred_second += metrics["second_order"]
            logger.cum_pred_full += metrics["full_law"]

            
            delta_actual = curr_L1_eval - prev_L1_eval

            stat_shift = 0.0 
            
            logger.cum_actual += delta_actual
            prev_L1_eval = curr_L1_eval

            # ----- D. Logging -----
            row = {
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
                "delta_L1_actual": delta_actual,
                "cum_actual": logger.cum_actual
            }
            logger.log(row)

    logger.close()
    print(f"\nExperiment complete. Data saved to: {results_path}")

# -----------------------------
# Full Experiment Setup
# -----------------------------
def run_vit_experiment(dataset,results_path, config):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Task loaders from the pre-instantiated dataset
    task1_loader, task2_loader = dataset.get_task_loaders(config["batch_size"])
    
    # Load HVP-safe proper ViT backbone
    model = get_proper_vit_tiny(num_classes=dataset.num_classes).to(device)

    # # 1. Train and save Task 1 basin
    train_task1_to_convergence(
        model, 
        task1_loader,  
        config["task1_lr"], 
        device,
        checkpoint_dir=config["checkpoint_dir"],
        max_epochs=config["task1_epochs"],
    )
    
    # checkpoint_dir='checkpoints'
    # path = os.path.join(checkpoint_dir, "vit_task1_basin.pth")
    # checkpoint = torch.load(path, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    # 2. Track ODE destruction on Task 2
    train_task2(
        model, 
        task1_loader, 
        task2_loader, 
        config["results_path"], 
        config["task2_epochs"], 
        config["task2_lr"], 
        config["log_interval"], 
        config["buffer_batches"], 
        config['max_steps'],
        device
    )