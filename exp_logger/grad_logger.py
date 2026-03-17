import torch
import os


class GradientLogger:

    def __init__(self, save_dir, interval=10):

        self.save_dir = save_dir
        self.interval = interval

        os.makedirs(save_dir, exist_ok=True)

    def maybe_save(self, model, step):

        if step % self.interval != 0:
            return

        grads = []

        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().flatten())

        g = torch.cat(grads)

        path = os.path.join(self.save_dir, f"grad_{step}.pt")

        torch.save(g.cpu(), path)