import torch
from PIL import Image
from nets import ResNet
from torchvision import transforms


def predict_res(model_path, model_name, img_path1, img_path2, img_path3, img_path4):
    device = torch.device("cuda")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if model_name == "resnet":
        model = ResNet(num_classes=2).to(device)
    else:
        raise ValueError("model name must be resnet!")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    label_names = ["Luminal A", "others"]
    image1 = torch.unsqueeze(transform(Image.open(img_path1).convert("RGB")), dim=0).to(device)
    image2 = torch.unsqueeze(transform(Image.open(img_path2).convert("RGB")), dim=0).to(device)
    image3 = torch.unsqueeze(transform(Image.open(img_path3).convert("RGB")), dim=0).to(device)
    image4 = torch.unsqueeze(transform(Image.open(img_path4).convert("RGB")), dim=0).to(device)
    with torch.no_grad():
        pred = torch.argmax(model(image1, image2, image3, image4), dim=-1).cpu().numpy()[0]

    print(f"The subtype prediction is:{label_names[pred]}")

    return label_names[pred]









