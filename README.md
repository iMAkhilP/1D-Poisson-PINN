# 🧠 Physics-Informed Neural Network (PINN) for 1D Poisson Equation

This repository demonstrates a simple yet powerful implementation of a **Physics-Informed Neural Network (PINN)** using **PyTorch** to solve the **1D Poisson Equation**, often associated with steady-state **heat conduction problems**.

## 📘 Problem Statement

We solve the following **1D Poisson equation**:

```
d²u(x)/dx² = f(x),   for x in (0, 1)
```

with **Dirichlet boundary conditions**:

```
u(0) = u(1) = 0
```

where `f(x)` is a known source term.

This equation describes, for example, the steady-state distribution of temperature in a rod with fixed temperature ends and an internal heat source.

## 🚀 Approach

We approximate the solution `u(x)` using a fully-connected neural network and train it by minimizing the **physics residual** (derived from the PDE) and the **boundary loss**. The model is trained **without any labeled data** — only the governing physics and boundary conditions guide the learning.

### Key Components:
- **PyTorch-based neural network (FCNet)**
- **Automatic differentiation** to compute `d²u/dx²`
- **Loss function** combining PDE residual and boundary condition loss
- **Adam optimizer** for training

## 🛠️ Model Architecture

```python
class FCNet(nn.Module):
    def __init__(self, layers):
        super(FCNet, self).__init__()
        self.net = nn.Sequential()
        for i in range(len(layers)-1):
            self.net.add_module(f"linear_{i}", nn.Linear(layers[i], layers[i+1]))
            if i < len(layers)-2:
                self.net.add_module(f"tanh_{i}", nn.Tanh())

    def forward(self, x):
        return self.net(x)
```

## 📈 Results

After training, the network learns to approximate the analytical solution to the Poisson equation. The learned function `u(x)` satisfies both the boundary conditions and the PDE.

## 📦 Files

- `1D_Poisson_PINN.ipynb`: Jupyter Notebook with full implementation.
- `README.md`: Project documentation.
---

Feel free to fork, modify, and build upon this to solve other PDEs!
