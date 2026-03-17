import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import random
import numpy as np
from models.gnn import GCNBaseline
from datasets_our.graph_dataset import ArxivDomainSplit
import torch_geometric

torch.serialization.add_safe_globals([
    torch_geometric.data.Data,
    torch_geometric.data.data.DataEdgeAttr,
    torch_geometric.data.data.DataTensorAttr,
    torch_geometric.data.storage.GlobalStorage
])

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Crucial for 1.0000 correlation in GNNs (Sparse operations can be non-deterministic)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def stabilize_seeds(seeds=[42, 123, 999, 2026, 7]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = ArxivDomainSplit()
    data, t1_mask, _ = loader.get_task_data()
    data = data.to(device)
    
    os.makedirs("checkpoints", exist_ok=True)

    for seed in seeds:
        print(f"\n--- Stabilizing Seed {seed} ---")
        set_seed(seed)
        
        # Architecture: 3-layer GCN, 256-hidden, No BatchNorm for 'Naked' manifold
        model = GCNBaseline(loader.num_features, 256, loader.num_classes, 3).to(device)
        
        # Using Adam for stabilization (to find the local minima), 
        # but your ODE tracking later will use Vanilla SGD for 1.0000 precision.
        optimizer = optim.Adam(model.parameters(), lr=0.001) 
        
        model.train()
        for epoch in range(500): # Extended epochs for deep basin stabilization
            optimizer.zero_grad()
            # Forward call: pass (x, edge_index) as per model_forward_functional
            out = model(data.x, data.edge_index) 
            loss = F.cross_entropy(out[t1_mask], data.y.squeeze()[t1_mask])
            loss.backward()
            optimizer.step()
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

        save_path = f"checkpoints/ogbn_arxiv_gcn_seed{seed}.pth"
        torch.save({
            'seed': seed,
            'model_state_dict': model.state_dict(),
            't1_mask': t1_mask # Saving mask ensures Task 1 anchor is identical
        }, save_path)
        print(f"Seed {seed} Basin Stabilized.")

if __name__ == "__main__":
    stabilize_seeds()