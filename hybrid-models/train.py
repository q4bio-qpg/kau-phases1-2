import torch
from torch import nn
from tqdm import tqdm


def train(model, train_dataloader, val_dataloader, epochs=50, model_type='r'):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_mses = []
    val_mses = []
    
    print("Starting Training for {} epochs".format(epochs))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    
    # Choose the criterion based on model type
    criterion = nn.MSELoss() if model_type == 'r' else nn.BCELoss()
    # criterion = nn.L1Loss()
    model.to(device)
    
    def fidelity_loss(output1, output2, target):
        # fidelity = 1 - torch.abs(torch.sum(output1 * torch.conj(output2), dim=-1))
        fidelity = torch.abs(torch.sum(output1 * torch.conj(output2), dim=-1))
        fidelity = torch.clamp(fidelity, 0, 1)
        
        fidelity = torch.exp(-3*fidelity)

        # target = torch.clamp(target, 0, 1)
        # target = torch.exp(3*target - 3)
        # print(fidelity, target)
        loss = criterion(fidelity, target)
        # print(loss)
        return loss
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_mse = 0.0  # Initialize MSE for the training phase

        # Store outputs and targets of the last epoch
        if epoch == epochs - 1:
            train_outputs = []
            train_targets = []
            val_outputs = []
            val_targets = []

        with tqdm(train_dataloader, unit="batch", desc=f"Epoch {epoch+1}/{epochs} (Training)") as tepoch:
            for inputs, labels in tepoch:
                input1, input2 = inputs
                input1 = torch.unsqueeze(input1, 2).to(device)
                input2 = torch.unsqueeze(input2, 2).to(device)
                labels = labels.clone().detach().float().to(device)

                optimizer.zero_grad()
                output1 = model(input1)
                output2 = model(input2)
                loss = fidelity_loss(output1, output2, labels)
                loss.backward()
                optimizer.step()

                fidelity_value = torch.abs(torch.sum(output1 * torch.conj(output2), dim=-1))
                
                # Track predictions and accuracy
                if model_type == 'r':
                    mse = torch.mean((fidelity_value - labels) ** 2).item()
                    # mse = torch.mean((torch.exp(-3 * fidelity_value) - labels) ** 2).item()
                    train_mse += mse * len(labels)
                else:
                    predicted = ((1 - fidelity_value) > 0.5).float()
                    acc = (predicted == labels).sum().item() / len(labels)
                    train_acc += acc * len(labels)

                train_loss += loss.item() * len(labels)

                if epoch == epochs - 1:
                    train_outputs.append(fidelity_value.cpu())
                    # train_outputs.append(torch.exp(-3 * fidelity_value).cpu())
                    train_targets.append(labels.cpu())

                tepoch.set_postfix(loss=loss.item(), accuracy=acc if model_type != 'r' else None, mse=mse if model_type == 'r' else None)
        
        val_loss = 0.0
        val_acc = 0.0
        val_mse = 0.0  # Initialize MSE for the validation phase
        # if epoch == epochs - 1:
        # if True:
        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()

            with torch.no_grad():
                with tqdm(val_dataloader, unit="batch", desc=f"Epoch {epoch+1}/{epochs} (Validation)") as veepoch:
                    for inputs, labels in veepoch:
                        input1, input2 = inputs
                        input1 = torch.unsqueeze(input1, 2).to(device)
                        input2 = torch.unsqueeze(input2, 2).to(device)
                        labels = labels.clone().detach().float().to(device)

                        output1 = model(input1)
                        output2 = model(input2)
                        loss = fidelity_loss(output1, output2, labels)

                        fidelity_value = torch.abs(torch.sum(output1 * torch.conj(output2), dim=-1))

                        if model_type == 'r':
                            # mse = torch.mean((fidelity_value - labels) ** 2).item()
                            mse = torch.mean((torch.exp(-3 * fidelity_value) - labels) ** 2).item()
                            val_mse += mse * len(labels)
                        else:
                            predicted = ((1 - fidelity_value) > 0.5).float()
                            # labels = (labels > 0.5)
                            acc = (predicted == (labels > 0.5)).sum().item() / len(labels)
                            val_acc += acc * len(labels)

                        val_loss += loss.item() * len(labels)

                        if epoch == epochs - 1:
                            val_outputs.append(fidelity_value.cpu())
                            # val_outputs.append(torch.exp(-3 * fidelity_value).cpu())
                            val_targets.append(labels.cpu())

                        veepoch.set_postfix(val_loss=loss.item(), val_accuracy=acc if model_type != 'r' else None, val_mse=mse if model_type == 'r' else None)

        train_loss /= len(train_dataloader.dataset)
        val_loss /= len(val_dataloader.dataset)
        
        # Average the accuracies or MSE depending on model type
        if model_type == 'r':
            train_mse /= len(train_dataloader.dataset)
            val_mse /= len(val_dataloader.dataset)
            train_mses.append(train_mse)
            val_mses.append(val_mse)
        else:
            train_acc /= len(train_dataloader.dataset)
            val_acc /= len(val_dataloader.dataset)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Training Loss: {train_loss:.4f}, "
              f"Training {'MSE' if model_type == 'r' else 'Accuracy'}: {train_mse if model_type == 'r' else train_acc:.4f} | "
              f"Validation Loss: {val_loss:.4f}, "
              f"Validation {'MSE' if model_type == 'r' else 'Accuracy'}: {val_mse if model_type == 'r' else val_acc:.4f}")

    # Concatenate outputs and targets for the last epoch
    train_outputs = torch.cat(train_outputs) if epochs > 0 else None
    train_targets = torch.cat(train_targets) if epochs > 0 else None
    val_outputs = torch.cat(val_outputs) if epochs > 0 else None
    val_targets = torch.cat(val_targets) if epochs > 0 else None

    if model_type == 'r':
        return train_losses, train_mses, val_losses, val_mses, train_outputs, train_targets, val_outputs, val_targets
    else:
        return train_losses, train_accuracies, val_losses, val_accuracies, train_outputs, train_targets, val_outputs, val_targets