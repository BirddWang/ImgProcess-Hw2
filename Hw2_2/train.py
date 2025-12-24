from model import ResNet18
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from train_utils import load_data, draw_curve


def train(model, device, epochs, optimizer, criterion, train_loader, val_loader):
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

    return train_acc_list, train_loss_list, val_acc_list, val_loss_list


def arg_parse():
    parser = argparse.ArgumentParser(description='ResNet18 Training')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 15)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_data(args.batch_size)

    model = ResNet18().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_acc_list, train_loss_list, val_acc_list, val_loss_list = train(model, device, args.epochs, optimizer, criterion, train_loader, test_loader)
    draw_curve(train_loss_list, train_acc_list, val_loss_list, val_acc_list)
    
    torch.save(model.state_dict(), f'Hw2_2/model/weight.pth')

if __name__ == '__main__':
    main()