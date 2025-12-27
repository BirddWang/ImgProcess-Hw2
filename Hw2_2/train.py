from model import ResNet18
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from train_utils import load_data, draw_curve
import os


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, verbose=False, delta=0.0, best_model_path='best_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.best_model_path = best_model_path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.best_model_path)
        self.val_loss_min = val_loss


def train(model, device, epochs, optimizer, scheduler, criterion, train_loader, val_loader, early_stopping=None):
    train_acc_list = []
    val_acc_list = []
    train_loss_list = []
    val_loss_list = []

    for epoch in range(1, epochs + 1):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            if batch_idx % 100 == 0 and batch_idx > 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
                train_loss = running_loss / len(train_loader)
                train_accuracy = 100. * correct / total
                train_acc_list.append(train_accuracy)
                train_loss_list.append(train_loss)
                running_loss = 0.0
                correct = 0
                total = 0
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == target).sum().item()
        val_loss /= len(val_loader)
        val_accuracy = 100. * correct / len(val_loader.dataset)
        val_acc_list.append(val_accuracy)
        val_loss_list.append(val_loss)
        print(f'\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({val_accuracy:.2f}%)\n')

        # Learning rate scheduling
        scheduler.step()

        # Early stopping
        if early_stopping is not None:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    return train_acc_list, train_loss_list, val_acc_list, val_loss_list


def arg_parse():
    parser = argparse.ArgumentParser(description='ResNet18 Training')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train (default: 25)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay for AdamW (default: 5e-4)')
    parser.add_argument('--early-stop-patience', type=int, default=7, help='patience for early stopping (default: 7)')
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()

    # Device detection with MPS support for Apple Silicon
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = "CUDA"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple Metal Performance Shaders (MPS)"
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    print(f"Using device: {device_name}")

    train_loader, test_loader = load_data(args.batch_size)

    model = ResNet18().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # Create best model save path
    best_model_path = 'Hw2_2/model/best_model.pth'
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    early_stopping = EarlyStopping(patience=args.early_stop_patience, verbose=True, best_model_path=best_model_path)

    train_acc_list, train_loss_list, val_acc_list, val_loss_list = train(model, device, args.epochs, optimizer, scheduler, criterion, train_loader, test_loader, early_stopping)

    # Load the best model weights before saving
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    draw_curve(train_loss_list, train_acc_list, val_loss_list, val_acc_list)

    torch.save(model.state_dict(), f'Hw2_2/model/weight.pth')

if __name__ == '__main__':
    main()