import torch
import torch.nn.functional as F

class ManualFC:
    def __init__(
        self, X: torch.Tensor, Y: torch.Tensor, 
    ) -> None:
        (resolution, ) = X.shape
        assert (X[1:] - X[:-1] > 0).all()
        self.bias = (- X).float()
        self.weight = torch.zeros_like(Y).float()
        self.final_bias = Y[0]
        for i in range(resolution - 1):
            self.weight[i] = (Y[i + 1] - self(X[i + 1])) / (
                X[i + 1] - X[i]
            )
    
    def __call__(self, x: torch.Tensor):
        x = x.unsqueeze(-1) + self.bias
        x = F.relu(x)
        x = (self.weight * x).sum(dim=-1)
        x = x + self.final_bias
        return x
