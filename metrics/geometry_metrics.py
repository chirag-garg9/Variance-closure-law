import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, jvp

def flatten_grads_func(grads_dict):
    return torch.cat([g.reshape(-1) for g in grads_dict.values() if g is not None])

def get_logits(out):
    if hasattr(out, "logits"): return out.logits
    if isinstance(out, (tuple, list)): return out[0]
    return out

def model_forward_functional(model, p_dict, inputs):
    """
    General functional wrapper for torch.func transformations.
    Ensures GNN tuples are unpacked correctly for functional_call.
    """
    if isinstance(inputs, tuple):
        # Unpack (x, edge_index, mask) -> pass (x, edge_index) to model
        return functional_call(model, p_dict, (inputs[0], inputs[1]))
    return functional_call(model, p_dict, (inputs,))

def get_loss_dispatch(logits, labels, x_data=None):
    """Unified Loss Dispatcher supporting GNN masking."""
    # GNN Case: Use the mask stored in the 3rd index of the tuple
    if isinstance(x_data, tuple) and len(x_data) > 2:
        mask = x_data[2]
        return F.cross_entropy(logits[mask], labels)
    
    # Causal LLM Case
    if logits.dim() == 3:
        return F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)), 
            labels[:, 1:].contiguous().view(-1)
        )
    
    # Standard Vision/MLP Case
    return F.cross_entropy(logits, labels)

def compute_geometry(model, x1, y1, x2, y2, lr):
    # Only track parameters with requires_grad=True (Supports LoRA/Freeze)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    
    def loss_fn(p_dict, x_data, y):
        out = model_forward_functional(model, p_dict, x_data)
        logits = get_logits(out)
        return get_loss_dispatch(logits, y, x_data)

    # Gradient Calculation (Double Precision for 1.0 Correlation)
    g1_local_dict = grad(loss_fn)(params, x1, y1)
    g1_local_flat = flatten_grads_func(g1_local_dict).to(torch.float64)
    
    g2_dict = grad(loss_fn)(params, x2, y2)
    g2_flat = flatten_grads_func(g2_dict).to(torch.float64)

    # 1st Order Alignment (A)
    A = torch.dot(g1_local_flat, g2_flat)

    # 2nd Order Curvature (C) -> H1 * g2
    _, h1_g2 = jvp(lambda p: grad(loss_fn)(p, x1, y1), (params,), (g2_dict,))
    h1_g2_flat = flatten_grads_func(h1_g2).to(torch.float64)
    C = torch.dot(g2_flat, h1_g2_flat)

    # 2nd Order Cross-Rotation (X) -> H2 * g1
    _, h2_g1 = jvp(lambda p: grad(loss_fn)(p, x2, y2), (params,), (g1_local_dict,))
    h2_g1_flat = flatten_grads_func(h2_g1).to(torch.float64)
    X = torch.dot(g2_flat, h2_g1_flat) 

    return {
        "A": A.item(), "C": C.item(), "X": X.item(),
        "first_order": (-lr * A).item(),
        "second_order": ((-lr * A + 0.5 * (lr**2) * C).item()),
        "full_law": (-lr * A + 0.5 * (lr**2) * (C + X)).item(),
        "norm_g1": torch.norm(g1_local_flat).item(),
        "norm_g2": torch.norm(g2_flat).item(),
        "kappa": (C / (torch.norm(g2_flat)**2 + 1e-12)).item()
    }

def tracking_step(model, optimizer, fixed_x1, fixed_y1, x2, y2, lr, prev_L1_eval,step=None,grad_logger=None):
    """Universal tracking loop wrapper with proper GNN tuple unpacking."""
    stat_shift = 0.0

    # 1. Geometry Calculation (The Taylor/ODE Terms)
    metrics = compute_geometry(model, fixed_x1, fixed_y1, x2, y2, lr)

    # 2. Parameter Update (The Physical Move)
    optimizer.zero_grad()
    
    # FIXED: Dispatching for the standard model() call
    if isinstance(x2, tuple):
        # Unpack: model(x, edge_index)
        out2 = model(x2[0], x2[1])
    else:
        out2 = model(x2)
        
    model.train()
    loss2 = get_loss_dispatch(get_logits(out2), y2, x2)
    loss2.backward()
    if grad_logger:
        grad_logger.maybe_save(model,step)
    optimizer.step()

    # 3. Actual Drift Measurement (The Target)
    model.eval()
    with torch.no_grad():
        if isinstance(fixed_x1, tuple):
            out1 = model(fixed_x1[0], fixed_x1[1])
        else:
            out1 = model(fixed_x1)
            
        curr_L1_eval = get_loss_dispatch(get_logits(out1), fixed_y1, fixed_x1).item()
    
    actual_drift = curr_L1_eval - prev_L1_eval
    return metrics, actual_drift, stat_shift, curr_L1_eval