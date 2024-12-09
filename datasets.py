import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split


def get_datasets(root="data"):
    df = pd.read_excel(root + "/" + "train_metadata.xlsx", sheet_name=0)
    df["label"] = df["subtype"].apply(lambda x: 0 if x == "Luminal A" else 1)
    names = df["patient_id"].values.tolist()
    x_train, x_test, y_train, y_test = train_test_split(names, df["label"].values.tolist(), train_size=0.8, shuffle=True)
    with open("data/train.txt", "w", encoding="utf-8") as f1, open("data/test.txt", "w", encoding="utf-8") as f2:
        for i in range(len(x_train)):
            f1.write(x_train[i] + "\t" + str(y_train[i]) + "\n")
        for i in range(len(x_test)):
            f2.write(x_test[i] + "\t" + str(y_test[i]) + "\n")


class DataGenerator(Dataset):

    def __init__(self, root):
        super(DataGenerator, self).__init__()
        self.root = root
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.img_paths, self.labels = self.get_datasets()

    def __getitem__(self, item):
        img_name = self.img_paths[item]
        img_path1, img_path2, img_path3, img_path4 = f"pngs/{img_name}_1-1.dcm.png", f"pngs/{img_name}_1-2.dcm.png", f"pngs/{img_name}_1-3.dcm.png", f"pngs/{img_name}_1-4.dcm.png"
        img1 = Image.open(img_path1).convert("RGB")
        img2 = Image.open(img_path2).convert("RGB")
        img3 = Image.open(img_path3).convert("RGB")
        img4 = Image.open(img_path4).convert("RGB")

        return torch.FloatTensor(self.transforms(img1)), torch.FloatTensor(self.transforms(img2)), torch.FloatTensor(self.transforms(img3)), torch.FloatTensor(self.transforms(img4)), torch.from_numpy(np.array(self.labels[item])).long()

    def __len__(self):
        return len(self.labels)

    def get_datasets(self):
        img_paths, labels = [], []
        with open(self.root, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line_split = line.strip().split("\t")
                img_paths.append(line_split[0])
                labels.append(int(line_split[-1]))

        return img_paths, labels


if __name__ == '__main__':
    get_datasets()
