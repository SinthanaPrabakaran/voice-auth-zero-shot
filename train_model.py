import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from your_dataset import VoiceDataset
from your_model import CNNModel
from utils import calculate_accuracy

# STEP 1: Initialize WandB
wandb.init(project="voice-auth-cnn", name="cnn-training-run-1")

config = {
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001,
}
wandb.config.update(config)

# STEP 2: Data
train_dataset = VoiceDataset("audio_samples/train")
val_dataset = VoiceDataset("audio_samples/val")
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

# STEP 3: Model + Loss + Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# STEP 4: Training
for epoch in range(config["epochs"]):
    model.train()
    total_loss = 0
    total_acc = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        acc = calculate_accuracy(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc

    avg_loss = total_loss / len(train_loader)
    avg_acc = total_acc / len(train_loader)

    # STEP 5: Validation
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels)

            val_loss += loss.item()
            val_acc += acc

    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = val_acc / len(val_loader)

    # STEP 6: Log to WandB
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        "train_accuracy": avg_acc,
        "val_loss": avg_val_loss,
        "val_accuracy": avg_val_acc,
    })

    print(f"✅ Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Acc={avg_val_acc:.4f}")

# STEP 7: Save model
torch.save(model.state_dict(), "models/cnn_model.pth")
wandb.save("models/cnn_model.pth")
