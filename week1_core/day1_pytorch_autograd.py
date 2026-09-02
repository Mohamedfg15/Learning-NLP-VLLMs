import torch
import torch.nn as nn

###########################
#Configuration
###########################

Learning_rate=0.1
epochs=100

#############################
#Dataset
#############################

x_train=torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0]
])
y_train=torch.tensor([
    [3.0],
    [5.0],
    [7.0],
    [9.0]
])

x_val=torch.tensor([
    [5.0],
    [6.0]
])
y_val=torch.tensor([
    [11.0],
    [13.0]
])
#####################
#Model
####################
class mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(1,1)
    def forward(self,x):
        return self.linear(x)
#####################
#training function
####################
def train(model,x,y,loss_f,optimizer,epochs):
    for epoch in range(epochs):
        optimizer.zero_grad()
        prediction=model(x)
        loss=loss_f(prediction,y)
        loss.backward()
        optimizer.step()
        if epoch %10==0:
            print(f"{epoch}/{epochs}"
                  f"{loss.item() :.4f}"
                  )
###########################
#validation
###########################
def validation(model,x,y,loss_fn):
    model.eval()
    with torch.no_grad():
        prediction=model(x)
        loss=loss_fn(prediction,y)
    print(f"validation_loss : {loss.item():.4f}")
    return loss.item()
######################
#main
#####################
def main():

    # Create Model
    model = mymodel()

    # Loss Function
    loss_fn = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=Learning_rate
    )

    # Training
    train(
        model,
        x_train,
        y_train,
        loss_fn,
        optimizer,
        epochs
    )

    # Validation
    validation(
        model,
        x_val,
        y_val,
        loss_fn
    )

    # Show learned parameters
    print("\nLearned Parameters:")

    for name, param in model.named_parameters():
        print(f"{name}: {param.data}")


if __name__ == "__main__":
    main()