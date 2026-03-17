import torch

class Task1GradientBuffer:
    def __init__(self, model, loader, buffer_batches=4, device="cpu"):
        self.model = model
        self.loader = loader
        self.buffer_batches = buffer_batches
        self.device = device
        self.iterator = iter(loader)

    def flatten(self, grads):
        # CRITICAL FIX: Safely ignore None gradients
        return torch.cat([g.reshape(-1) for g in grads if g is not None])

    def compute_grad(self, loss):
        # CRITICAL FIX: No create_graph or retain_graph.
        # We only want the physical vector, not the derivative history.
        grads = torch.autograd.grad(
            loss,
            self.model.parameters(),
            retain_graph=False,
            create_graph=False
        )
        # Detach immediately to prevent any accidental graph leakage
        return self.flatten(grads).detach()

    def estimate_g1(self, criterion):
        g_accum = None
        
        # CRITICAL FIX: Freeze model state. 
        # We are measuring the landscape, we cannot allow BatchNorm 
        # to update its running statistics during the probe.
        was_training = self.model.training
        self.model.eval()

        for _ in range(self.buffer_batches):
            try:
                x, y = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.loader)
                x, y = next(self.iterator)

            x = x.to(self.device)
            y = y.to(self.device)

            out = self.model(x)
            loss = criterion(out, y)

            g = self.compute_grad(loss)

            if g_accum is None:
                g_accum = g
            else:
                g_accum += g

        # Restore the model to its original training state
        if was_training:
            self.model.train()

        # Return the exact, detached, averaged g1 vector
        return g_accum / self.buffer_batches