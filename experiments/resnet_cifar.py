# Auto-generated file
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Import the custom HVP-safe ResNet and instant-split CIFAR
from models.resnet import ResNet20
from datasets_our.cifar_split import CIFAR10Split

# Import the bulletproof measurement tools
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

def train_task1_to_convergence(model, loader, lr, device, max_epochs=200, tolerance=1e-4, patience=3):
    # We use Adam here just to force the network into the local minimum quickly.
    # The geometry isn't measured yet, so the optimizer choice here doesn't corrupt the math.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    print(f"Beginning Task 1 Training (Auto-Converging up to {max_epochs} epochs)...")
    
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
            
            # THE THEORY FLEX: Track the steepness of the manifold before updating
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(loader)
        loss_change = best_loss - epoch_loss
        
        print(f"Task 1 Epoch {epoch+1:02d} | Loss: {epoch_loss:.4f} | Δ Loss: {loss_change:.6f} | ||∇L1||: {grad_norm:.4f}")
        
        # --- CONVERGENCE CHECK ---
        if abs(loss_change) < tolerance:
            patience_counter += 1
            print(f"  -> Basin flatline detected. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\n[CONVERGENCE REACHED] Manifold geometry stabilized at Epoch {epoch+1}.")
                break
        else:
            if loss_change > 0:
                best_loss = epoch_loss
            patience_counter = 0
            
    return model


# -----------------------------
# Train Task 2 with continuous measurement
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
    # CRITICAL: We strictly use SGD here so the physical PyTorch step 
    # perfectly matches the theoretical Taylor step: w_{t+1} = w_t - \eta g_2
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    buffer = Task1GradientBuffer(model, task1_loader, buffer_batches, device)
    logger = CSVLogger(results_path)
    
    # 1. Isolate the Fixed Probe Batch (Prevents Mini-Batch Noise)
    fixed_x1, fixed_y1 = next(iter(task1_loader))
    fixed_x1, fixed_y1 = fixed_x1.to(device), fixed_y1.to(device)

    step = logger.last_step + 1
    # 2. Baseline Measurement (Strictly in eval() to prevent BatchNorm leakage)
    model.eval()
    with torch.no_grad():
        prev_L1 = criterion(model(fixed_x1), fixed_y1).item()

    print(f"\nBeginning Task 2 Continuous ODE Tracking...")
    step = 0
    grad_logger = GradientLogger(
        save_dir="grad_logs/experiment/resnet",
        interval=5   # save every 20 steps
    )
    for epoch in range(epochs):
        if step>max_steps: break
        for x2, y2 in tqdm(task2_loader, desc=f"Epoch {epoch}"):
            x2, y2 = x2.to(device), y2.to(device)
            step += 1

            # ----- A. ODE Kinematic Measurement -----
            
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
                step=step,
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
            prev_L1 = curr_L1_eval

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
def run_resnet_cifar_experiment(dataset,results_path,config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Initializing ResNet-20 on {device.upper()}...")
    task1_loader, task2_loader = dataset.get_task_loaders(config["batch_size"])

    # Output dimension is 10 for CIFAR-10
    model = ResNet20(num_classes=dataset.num_classes).to(device)

    train_task1_to_convergence(
        model=model,
        loader=task1_loader,
        max_epochs=config["task1_epochs"],
        lr=config["task1_lr"],
        device=device
    )

    train_task2(
        model=model,
        task1_loader=task1_loader,
        task2_loader=task2_loader,
        results_path=config["results_path"],
        epochs=config["task2_epochs"],
        lr=config["task2_lr"],
        log_interval=config["log_interval"],
        buffer_batches=config["buffer_batches"],
        max_steps=config['max_steps'],
        device=device
    )