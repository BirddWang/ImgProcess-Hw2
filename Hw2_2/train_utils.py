from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Resize((32, 32))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    print("size of train dataset:", len(train_dataset))
    print("size of test dataset:", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def draw_curve(train_loss_list, train_accuracy_list,
               test_loss_list, test_accuracy_list):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].plot(train_loss_list, label='Train Loss')
    axs[0, 0].set_title('Train Loss Curve')
    axs[0, 0].set_xlabel('Iterations')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_ylim(0, 1)
    axs[0, 0].legend()
    axs[0, 1].plot(train_accuracy_list, label='Train Accuracy', color='orange')
    axs[0, 1].set_title('Train Accuracy Curve')
    axs[0, 1].set_xlabel('Iterations')
    axs[0, 1].set_ylabel('Accuracy (%)')
    axs[0, 1].set_ylim(0, 100)
    axs[0, 1].legend()
    axs[1, 0].plot(test_loss_list, label='Validation Loss', color='green')
    axs[1, 0].set_xlabel('Iterations')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].set_ylim(0, 1)
    axs[1, 0].set_title('Validation Loss Curve')
    axs[1, 0].legend()
    axs[1, 1].plot(test_accuracy_list, label='Validation Accuracy', color='red')
    axs[1, 1].set_xlabel('Iterations')
    axs[1, 1].set_ylabel('Accuracy (%)')
    axs[1, 1].set_ylim(0, 100)
    axs[1, 1].set_title('Validation Accuracy Curve')
    axs[1, 1].legend()
    plt.tight_layout()
    plt.savefig('Hw2_2/Loss&Acc.png')
