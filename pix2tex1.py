import argparse
import torch
from torch import nn
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torchvision import transforms
import PIL.Image
import torchvision.transforms.functional as TF
import sys
import pandas as pd
import os


class LatexDataset(torch.utils.data.Dataset):
    """Dataset for LaTeX formula images from im2latex-100k dataset."""

    def __init__(self, annotations_file, img_dir):
        """
        Args:
            annotations_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
        """
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        image = PIL.Image.open(img_path)
        image = TF.to_tensor(image)
        label = self.img_labels.iloc[idx, 2]
        return image, label


class ResNetLatexModel(nn.Module):
    """Model to convert images of LaTeX formulas into LaTeX code using ResNet-50."""

    def __init__(self):
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        self.decoder = nn.Linear(1000, 512)

    def forward(self, x):
        features = self.resnet(x)
        output = self.decoder(features)
        return output


def parse_args():
    parser = argparse.ArgumentParser(description='Convert images of LaTeX formulas to LaTeX code.')
    parser.add_argument('image_path', type=str, help='Path to the image file.')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load model
    model = ResNetLatexModel()
    model.eval()

    # Load image
    image = PIL.Image.open(args.image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        prediction = model(image)
        predicted_code = torch.argmax(prediction, dim=1)
        print("Predicted LaTeX code:", predicted_code)


if __name__ == "__main__":
    main()
