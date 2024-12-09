import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from nets import ResNet
from datasets import DataGenerator
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


def train(model_name):
    device = torch.device("cuda")
    if model_name == "resnet":
        model = ResNet(num_classes=2).to(device)
    else:
        raise ValueError("model name must be resnet!")
    train_loader = DataLoader(DataGenerator(root="data/train.txt"), batch_size=8, shuffle=True)
    val_loader = DataLoader(DataGenerator(root="data/test.txt"), batch_size=16, shuffle=False)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    loss_func = nn.CrossEntropyLoss()
    train_accs, train_losses = [], []
    val_accs, val_losses = [], []
    max_val_acc = 0
    for epoch in range(20):
        train_acc, train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, loss_func, device, epoch)
        val_acc, val_loss = get_val_result(model, val_loader, loss_func, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f"model:{model_name},epoch:{epoch + 1},train acc:{train_acc},train loss:{train_loss},val acc:{val_acc},val loss:{val_loss}")
        if val_acc > max_val_acc:
            torch.save(model.state_dict(), f"models/{model_name}_best.pth")
            max_val_acc = val_acc
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"models/{model_name}_epoch{epoch + 1}.pth")

    plot_acc_loss(train_accs, train_losses, val_accs, val_losses, model_name)


def train_one_epoch(model, train_loader, optimizer, scheduler, loss_func, device, epoch):
    model.train()
    data = tqdm(train_loader)
    losses = []
    labels_true, labels_pred = np.array([]), np.array([])
    for batch, (x1, x2, x3, x4, y) in enumerate(data):
        labels_true = np.concatenate([labels_true, y.numpy()], axis=-1)
        x_train1, x_train2, x_train3, x_train4, labels_train = x1.to(device), x2.to(device), x3.to(device), x4.to(device), y.to(device)
        prob = model(x_train1, x_train2, x_train3, x_train4)
        pred = torch.argmax(prob, dim=-1).cpu().numpy()
        labels_pred = np.concatenate([labels_pred, pred], axis=-1)
        loss = loss_func(prob, labels_train)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        data.set_description_str(f"epoch:{epoch + 1},batch:{batch + 1},loss:{loss.item()},lr:{scheduler.get_last_lr()[0]:.7f}")
    scheduler.step()

    accuracy = accuracy_score(labels_true, labels_pred)
    losses = float(np.mean(losses))

    return accuracy, losses


def get_val_result(model, val_loader, loss_func, device):
    model.eval()
    data = tqdm(val_loader)
    labels_pred, labels_true = np.array([]), np.array([])
    labels_prob = []
    with torch.no_grad():
        for x1, x2, x3, x4, y in data:
            labels_true = np.concatenate([labels_true, y.numpy()], axis=-1)
            x_val1, x_val2, x_val3, x_val4, labels_val = x1.to(device), x2.to(device), x3.to(device), x4.to(device), y.to(device)
            prob = model(x_val1, x_val2, x_val3, x_val4)
            labels_prob.append(prob.cpu())
            pred = torch.argmax(prob, dim=-1).cpu().numpy()
            labels_pred = np.concatenate([labels_pred, pred], axis=-1)
    labels_prob = torch.cat(labels_prob, dim=0)
    accuracy = accuracy_score(labels_true, labels_pred)
    losses = loss_func(labels_prob, torch.from_numpy(labels_true).long())

    return accuracy, losses.item()


def plot_acc_loss(train_accs, train_losses, val_accs, val_losses, model_name):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_accs) + 1), train_accs, "r", label="train")
    plt.plot(range(1, len(val_accs) + 1), val_accs, "g", label="val")
    plt.title(f"{model_name}_accuracy-epoch")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_losses) + 1), train_losses, "r", label="train")
    plt.plot(range(1, len(val_losses) + 1), val_losses, "g", label="val")
    plt.title(f"{model_name}_loss-epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(f"images/{model_name}_epoch_acc_loss.jpg")


if __name__ == '__main__':
    train(model_name="resnet")
