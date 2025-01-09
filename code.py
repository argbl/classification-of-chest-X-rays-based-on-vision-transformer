from google.colab import drive
drive.mount('/content/drive')
import os
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch


train_data_path = '/content/drive/MyDrive/dataset/archive/train'
test_data_path = '/content/drive/MyDrive/dataset/archive/test'
save_model_path = '/content/drive/MyDrive/models'

os.makedirs(save_model_path, exist_ok=True)

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = ImageFolder(root=train_data_path, transform=transform)
test_dataset = ImageFolder(root=test_data_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Classes:", train_dataset.classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import ViTForImageClassification

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=len(train_dataset.classes)
)
model.train()
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()
class EarlyStopping:
    def __init__(self, patience=5, delta=0, save_path='checkpoint.pth'):
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            print(f"Model improved. Saved to {self.save_path}.")
        else:
            self.counter += 1
            print(f"Early stopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
def calculate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total
from tqdm import tqdm

def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, device, epochs, patience, save_path):
    early_stopping = EarlyStopping(patience=patience, save_path=save_path)

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        train_accuracy = calculate_accuracy(model, train_loader, device)
        train_accuracies.append(train_accuracy)

        model.eval()
        test_running_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).logits
                loss = criterion(outputs, labels)
                test_running_loss += loss.item()

        test_loss = test_running_loss / len(test_loader)
        test_losses.append(test_loss)
        test_accuracy = calculate_accuracy(model, test_loader, device)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss {train_loss:.4f}, Test Loss {test_loss:.4f}, Train Acc {train_accuracy:.4f}, Test Acc {test_accuracy:.4f}")

        early_stopping(test_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

    return train_losses, test_losses, train_accuracies, test_accuracies

epochs = 20
patience = 3
save_path = os.path.join(save_model_path, 'vit_best_model.pth')

train_losses, test_losses, train_accuracies, test_accuracies = train_and_evaluate(
    model, train_loader, test_loader, optimizer, criterion, device, epochs, patience, save_path
)
model.load_state_dict(torch.load(save_path))
print("Best model loaded.")
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.grid()
plt.show()
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def evaluate(model, data_loader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=class_names)
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    return report

class_names = train_dataset.classes
report = evaluate(model, test_loader, device, class_names)
print(report)
