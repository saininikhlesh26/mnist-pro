import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Configuration
LR = 0.0001
BATCH_SIZE = 32
EPOCHS = 5
MODEL_PATH = 'models/mnist_ann_model.pth'
PLOTS_DIR = 'plots'

# Ensure directories exist
os.makedirs('models', exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# 1. Dataset Preparation
def get_dataloaders():
    print("--- Loading MNIST Dataset ---")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Split into 50,000 training (as requested)
    train_indices = list(range(50000))
    train_subset = Subset(train_set, train_indices)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, test_set

# 2. Model Architecture (ANN)
class ANNModel(nn.Module):
    def __init__(self):
        super(ANNModel, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
            # CrossEntropyLoss applies Softmax internally
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

def train_model(model, train_loader, device):
    print(f"--- Training Model (Batch Size: {BATCH_SIZE}, LR: {LR}) ---")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    history = {'loss': [], 'accuracy': []}
    
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (batch_idx + 1) % 500 == 0:
                print(f"  Batch [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.4f}")
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        
    return history

def plot_training_history(history):
    print("--- Generating Training Plots ---")
    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss', color='#FF5733', lw=2)
    plt.title('Loss vs Epochs', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Train Accuracy', color='#3357FF', lw=2)
    plt.title('Accuracy vs Epochs', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/training_history.png')
    plt.close()

def evaluate_model(model, test_loader, device):
    print("--- Evaluating Model Performance ---")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted')
    rec = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"\nModel Evaluation Metrics:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'{PLOTS_DIR}/confusion_matrix.png')
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    train_loader, test_loader, _ = get_dataloaders()

    # 2. Build Model
    model = ANNModel().to(device)
    print("\nModel Summary (Total Parameters):", sum(p.numel() for p in model.parameters()))

    # 3. Train Model
    history = train_model(model, train_loader, device)

    # 4. Save Model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel state dictionary saved to {MODEL_PATH}")

    # 5. Plot History
    plot_training_history(history)

    # 6. Evaluate
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()
