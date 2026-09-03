import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Configuration
# ============================================================

LEARNING_RATE = 0.01
EPOCHS = 100
BATCH_SIZE = 2


# ============================================================
# Model
# ============================================================

class MyModel(nn.Module):
    """
    Simple linear model:
        y = wx + b
    """

    def __init__(self):
        super().__init__()

        self.w = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * self.w + self.b


# ============================================================
# Dataset
# ============================================================

class MyDataset(Dataset):
    """
    Simple Dataset for storing input and target tensors.
    """

    def __init__(self, x, y):
        super().__init__()

        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


# ============================================================
# Training
# ============================================================

def train(
    model,
    dataloader,
    optimizer,
    loss_fn,
    scaler,
    device,
    enable_amp,
    epochs
):
    """
    Train the model using mini-batches and Automatic
    Mixed Precision (AMP).
    """

    model.train()

    for epoch in range(epochs):

        for batch_x, batch_y in dataloader:

            # Move batch to the selected device
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Clear previous gradients
            optimizer.zero_grad()

            # Mixed Precision
            with torch.autocast(
                device_type=device.type,
                enabled=enable_amp
            ):
                prediction = model(batch_x)
                loss = loss_fn(prediction, batch_y)

            # Backpropagation with gradient scaling
            scaler.scale(loss).backward()

            # Update model parameters
            scaler.step(optimizer)
            scaler.update()

        # Log training progress
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Loss: {loss.item():.4f}"
            )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    x = torch.tensor([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [6.0]
    ])

    y = torch.tensor([
        [3.0],
        [5.0],
        [7.0],
        [9.0],
        [11.0],
        [13.0]
    ])

    dataset = MyDataset(x, y)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # --------------------------------------------------------
    # Device & AMP
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    enable_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        device="cuda",
        enabled=enable_amp
    )

    # --------------------------------------------------------
    # Model, Loss & Optimizer
    # --------------------------------------------------------

    model = MyModel().to(device)

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    train(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scaler=scaler,
        device=device,
        enable_amp=enable_amp,
        epochs=EPOCHS
    )

    # --------------------------------------------------------
    # Final Parameters
    # --------------------------------------------------------

    print("\nLearned Parameters:")

    for name, parameter in model.named_parameters():
        print(f"{name}: {parameter.item():.4f}")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()