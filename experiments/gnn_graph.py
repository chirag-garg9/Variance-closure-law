import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
from models.gnn import GCNBaseline
from metrics.geometry_metrics import tracking_step
from exp_logger.csv_logger import CSVLogger
import torch_geometric
from exp_logger.grad_logger import GradientLogger

torch.serialization.add_safe_globals([
    torch_geometric.data.Data,
    torch_geometric.data.data.DataEdgeAttr,
    torch_geometric.data.data.DataTensorAttr,
    torch_geometric.data.storage.GlobalStorage
])

def run_pretrained_gnn_experiment(dataset,results_path, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, t1_mask, t2_mask = dataset.get_task_data()
    data = data.to(device)
    y = data.y.squeeze()
    name, ext = os.path.splitext(config['checkpoint_path'])
    path = f"{name}_seed{config['seed']}{ext}"
    model = GCNBaseline(dataset.num_features, config['hidden_channels'], dataset.num_classes).to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    grad_logger = GradientLogger(
        save_dir="grad_logs/experiment/gnn",
        interval=5   # save every 20 steps
    )
    model.eval()
    with torch.no_grad():
        out_init = model(data.x, data.edge_index)
        prev_L1_eval = F.cross_entropy(out_init[t1_mask], y[t1_mask]).item()

    optimizer_sgd = optim.SGD(model.parameters(), lr=config['task2_lr'])
    logger = CSVLogger(config['results_path'])
    t2_indices = torch.where(t2_mask)[0]
    
    accum_pred = 0.0
    accum_actual = 0.0

    for step in tqdm(range(config['task2_steps']), desc="GNN ODE Track"):
        batch_idx = t2_indices[torch.randperm(len(t2_indices))[:1024]]
        
        metrics, actual_drift, stat_shift, curr_L1_eval = tracking_step(
            model=model, optimizer=optimizer_sgd,
            fixed_x1=(data.x, data.edge_index, t1_mask), fixed_y1=y[t1_mask],
            x2=(data.x, data.edge_index, batch_idx), y2=y[batch_idx],
            lr=config['task2_lr'], prev_L1_eval=prev_L1_eval,step = step,
            grad_logger = grad_logger
        )
        prev_L1_eval = curr_L1_eval
        logger.cum_pred_first += metrics["first_order"]
        logger.cum_pred_second += metrics["second_order"]
        logger.cum_pred_full += metrics["full_law"]
        logger.cum_actual += actual_drift


        if step % config['log_interval'] == 0:
            logger.log({
                    "step": step,
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
