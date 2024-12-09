import torch
import seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from nets import ResNet
from datasets import DataGenerator
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, roc_curve, confusion_matrix, \
    precision_recall_curve

rc = {'font.sans-serif': 'SimHei', 'axes.unicode_minus': False}
seaborn.set(context='notebook', style='ticks', rc=rc)


def one_hot(y, num_classes=3):
    y_ = np.zeros((len(y), num_classes))
    for i in range(len(y)):
        y_[i, int(y[i])] = 1

    return y_


def softmax(y):
    y_exp = np.exp(y)
    for i in range(len(y)):
        y_exp[i, :] = y_exp[i, :] / np.sum(y_exp[i, :])

    return y_exp


def get_test_result(model_path, model_name, columns, root="datasets/test.txt"):
    print(model_name)
    device = torch.device("cuda")
    if model_name == "resnet":
        model = ResNet(num_classes=2).to(device)
    else:
        raise ValueError("model name must be resnet!")
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    test_loader = DataLoader(DataGenerator(root=root), batch_size=32, shuffle=False)
    data = tqdm(test_loader)
    labels_true, labels_pred, labels_prob = np.array([]), np.array([]), []
    with torch.no_grad():
        for x1, x2, x3, x4, y in data:
            x_test1, x_test2, x_test3, x_test4 = x1.to(device), x2.to(device), x3.to(device), x4.to(device)
            prob = model(x_test1, x_test2, x_test3, x_test4)
            labels_prob.append(prob.cpu().numpy())
            pred = torch.argmax(prob, dim=-1).cpu().numpy()
            labels_pred = np.concatenate([labels_pred, pred], axis=-1)
            labels_true = np.concatenate([labels_true, y.numpy()], axis=-1)

    labels_prob = softmax(np.concatenate(labels_prob, axis=0))
    labels_onehot = one_hot(labels_true, num_classes=2)

    accuracy = accuracy_score(labels_true, labels_pred)
    precision = precision_score(labels_true, labels_pred)
    recall = recall_score(labels_true, labels_pred)
    f1 = f1_score(labels_true, labels_pred)
    print(f"accuracy:{accuracy},precision:{precision},recall:{recall},f1:{f1}")

    colors = ["green", "purple", "pink", "yellow"]
    plt.figure(figsize=(10, 10), dpi=300)
    plt.plot([0, 1], [0, 1], "r--")
    for i in range(len(columns)):
        fpr, tpr, _ = roc_curve(labels_onehot[:, i], labels_prob[:, i])
        plt.plot(fpr, tpr, colors[i], label=f"{columns[i]} AUC:{auc(fpr, tpr):.3f}", linewidth=3.0)

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("roc_curve")
    plt.legend()
    plt.savefig(f"images/{model_name.split('/')[-1].split('_')[0]}_roc_curve.jpg", dpi=300)

    plt.figure(figsize=(10, 10), dpi=300)
    for i in range(len(columns)):
        p, r, _ = precision_recall_curve(labels_onehot[:, i], labels_prob[:, i])
        plt.plot(p, r, colors[i], label=columns[i], linewidth=3.0)
    plt.xlabel("precision")
    plt.ylabel("recall")
    plt.legend()
    plt.savefig(f"images/{model_name.split('/')[-1].split('_')[0]}_pr_curve.jpg", dpi=300)

    matrix = pd.DataFrame(confusion_matrix(labels_true, labels_pred, normalize="true"), columns=columns, index=columns)
    plt.figure(figsize=(10, 10), dpi=300)
    seaborn.heatmap(matrix, annot=True, cmap="GnBu")
    plt.title("confusion_matrix")
    plt.savefig(f"images/{model_name.split('/')[-1].split('_')[0]}_confusion_matrix.jpg", dpi=300)


if __name__ == '__main__':
    label_names = ["Luminal A", "others"]
    get_test_result(model_path=f"models/resnet_best.pth", model_name="resnet", columns=label_names)

