import torch
from torch import Tensor
from torch.nn import LayerNorm, Linear, Module, ModuleList, Sequential, SiLU


class MessageLayer(Module):
    def __init__(self, dim_h: int) -> None:
        super().__init__()
        self.phi = Sequential(
            Linear(2 * dim_h + 1, dim_h),
            SiLU(),
            LayerNorm(dim_h),
            Linear(dim_h, dim_h),
            SiLU(),
        )

    def forward(self, hi: Tensor, hj: Tensor, d2: Tensor) -> Tensor:
        msg = torch.cat((hi, hj, d2), dim=-1)
        msg = self.phi(msg)
        return msg


class InvariantUpdateLayer(Module):
    def __init__(self, dim_h: int) -> None:
        super().__init__()
        self.message_layer = MessageLayer(dim_h)
        self.phi = Sequential(
            Linear(2 * dim_h, dim_h),
            SiLU(),
            Linear(dim_h, dim_h),
        )

    def forward(self, h: Tensor, e: Tensor, d2: Tensor) -> Tensor:
        # Step 1: get features of connected nodes
        ei = e[0]
        ej = e[1]
        hi = h[ei]
        hj = h[ej]

        # Step 2: calculate the message
        msg: Tensor = self.message_layer(hi, hj, d2)

        # Step 3: aggregate the message
        idx = ei.unsqueeze(-1).expand(msg.shape)
        agg = torch.scatter_add(torch.zeros_like(h), 0, idx, msg)

        # Step 4: apply last sequential layer
        new_h = torch.cat((h, agg), dim=-1)
        new_h = self.phi(new_h)
        new_h = h + new_h

        return new_h


class EquivariantUpdateLayer(Module):
    def __init__(self, dim_h: int) -> None:
        super().__init__()
        self.message_layer = MessageLayer(dim_h)
        self.phi = Sequential(
            Linear(dim_h, dim_h),
            SiLU(),
            Linear(dim_h, 1),
        )

    def forward(self, h: Tensor, x: Tensor, e: Tensor, dx: Tensor, d2: Tensor) -> Tensor:
        # Step 1: get features of connected nodes
        ei = e[0]
        ej = e[1]
        hi = h[ei]
        hj = h[ej]

        # Step 2: calculate the message
        msg = self.message_layer(hi, hj, d2)
        msg = dx * self.phi(msg)

        # Step 3: aggregate the message
        idx = ei.unsqueeze(-1).expand(msg.shape)
        agg = torch.scatter_add(torch.zeros_like(x), 0, idx, msg)

        # Step 4: add residual connection
        new_x = x + agg

        return new_x


class EGCL(Module):
    def __init__(self, dim_h: int, equivariant_update: bool = True, invariant_update: bool = True) -> None:
        super().__init__()
        if equivariant_update:
            self.equ_layer = EquivariantUpdateLayer(dim_h)
        else:
            self.equ_layer = None

        if invariant_update:
            self.inv_layer = InvariantUpdateLayer(dim_h)
        else:
            self.inv_layer = None

    def forward(self, h: Tensor, x: Tensor, e: Tensor) -> tuple[Tensor, Tensor]:
        # Step 1: get positions of connected nodes
        ei = e[0]
        ej = e[1]
        xi = x[ei]
        xj = x[ej]

        # Step 2: calculate pair distances
        dx = xi - xj
        d2 = torch.sum(dx ** 2, dim=-1, keepdim=True)
        dx = dx / (torch.sqrt(d2) + 1e2)

        # Step 3: update positions
        if self.equ_layer is not None:
            new_x = self.equ_layer(h, x, e, dx, d2)
        else:
            new_x = x

        # Step 3: update node features
        if self.inv_layer is not None:
            new_h = self.inv_layer(h, e, d2)
        else:
            new_h = h

        return new_h, new_x


class EGNN(Module):
    def __init__(self, dim_features: int, num_layers: int) -> None:
        super().__init__()
        if num_layers < 1: raise ValueError("num_layers must be greater than 0")
        layers = [EGCL(dim_features) for _ in range(num_layers - 1)]
        layers.append(EGCL(dim_features, invariant_update=False))
        self.layers = ModuleList(layers)

    def forward(self, h: Tensor, x: Tensor, e: Tensor) -> tuple[Tensor, Tensor]:
        for layer in self.layers:
            h, x = layer(h, x, e)

        return h, x

class CompleteLocalFrameEquivariantUpdateLayer(Module):
    def __init__(self, dim_h: int) -> None:
        super().__init__()
        self.message_layer = MessageLayer(dim_h)
        self.phi_u1 = Sequential(
            Linear(dim_h, dim_h),
            SiLU(),
            Linear(dim_h, 1),
        )
        self.phi_u2 = Sequential(
            Linear(dim_h, dim_h),
            SiLU(),
            Linear(dim_h, 1),
        )
        self.phi_u3 = Sequential(
            Linear(dim_h, dim_h),
            SiLU(),
            Linear(dim_h, 1),
        )

    def forward(self, h: Tensor, x: Tensor, e: Tensor, u1: Tensor, u2: Tensor, u3: Tensor, d2: Tensor) -> Tensor:
        # Step 1: get features of connected nodes
        ei = e[0]
        ej = e[1]
        hi = h[ei]
        hj = h[ej]

        # Step 2: calculate the message
        msg = self.message_layer(hi, hj, d2)
        msg = u1 * self.phi_u1(msg) + u2 * self.phi_u2(msg) + u3 * self.phi_u3(msg)

        # Step 3: aggregate the message
        idx = ei.unsqueeze(-1).expand(msg.shape)
        agg = torch.scatter_add(torch.zeros_like(x), 0, idx, msg)

        # Step 4: add residual connection
        new_x = x + agg

        return new_x


class CompleteLocalFrameEGCL(Module):
    def __init__(self, dim_h: int, equivariant_update: bool = True, invariant_update: bool = True) -> None:
        super().__init__()
        if equivariant_update:
            self.equ_layer = CompleteLocalFrameEquivariantUpdateLayer(dim_h)
        else:
            self.equ_layer = None

        if invariant_update:
            self.inv_layer = InvariantUpdateLayer(dim_h)
        else:
            self.inv_layer = None

    def forward(self, h: Tensor, x: Tensor, e: Tensor) -> tuple[Tensor, Tensor]:
        # Step 1: get positions of connected nodes
        ei = e[0]
        ej = e[1]
        xi = x[ei]
        xj = x[ej]

        # Step 2: calculate vectors
        dx = xi - xj
        d2 = torch.sum(dx ** 2, dim=-1, keepdim=True)
        u1 = dx / (torch.sqrt(d2) + 1e-8)
        u2 = torch.cross(xi, xj, dim=-1)
        u2 = u2 / (torch.sqrt(torch.sum(u2 ** 2, dim=-1, keepdim=True)) + 1e-8)
        u3 = torch.cross(u1, u2, dim=-1)
        u3 = u3 / (torch.sqrt(torch.sum(u3 ** 2, dim=-1, keepdim=True)) + 1e-8)

        # Step 3: update positions
        if self.equ_layer is not None:
            new_x = self.equ_layer(h, x, e, u1, u2, u3, d2)
        else:
            new_x = x

        # Step 3: update node features
        if self.inv_layer is not None:
            new_h = self.inv_layer(h, e, d2)
        else:
            new_h = h

        return new_h, new_x


class CompleteLocalFrameEGNN(Module):
    def __init__(self, dim_features: int, num_layers: int) -> None:
        super().__init__()
        if num_layers < 1: raise ValueError("num_layers must be greater than 0")
        layers = [CompleteLocalFrameEGCL(dim_features) for _ in range(num_layers - 1)]
        layers.append(CompleteLocalFrameEGCL(dim_features, invariant_update=False))
        self.layers = ModuleList(layers)

    def forward(self, h: Tensor, x: Tensor, e: Tensor) -> tuple[Tensor, Tensor]:
        # Remove center
        m = torch.mean(x, dim=0, keepdim=True)
        c = x - m
        # Forward pass
        for layer in self.layers:
            h, c = layer(h, c, e)
        # Add back center
        x = c + m

        return h, x