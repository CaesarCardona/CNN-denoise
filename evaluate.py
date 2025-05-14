import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# --- CNN model definition ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 3, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.upsample(x)
        x = self.relu(self.conv5(x))
        x = self.upsample(x)
        x = self.conv6(x)
        return x

# --- Dataset class ---
class NoisyCleanDataset(torch.utils.data.Dataset):
    def __init__(self, noisy_dir, clean_dir, transform=None):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.image_files = [f for f in os.listdir(noisy_dir) if f.lower().endswith(('.jpg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        noisy_path = os.path.join(self.noisy_dir, filename)
        clean_path = os.path.join(self.clean_dir, filename)

        noisy = Image.open(noisy_path).convert('RGB')
        clean = Image.open(clean_path).convert('RGB')

        if self.transform:
            noisy = self.transform(noisy)
            clean = self.transform(clean)

        return noisy, clean, filename

# --- Utility function ---
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# --- Config ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

noisy_dir = 'dataset-with-noise'
clean_dir = 'dataset'
output_dir = 'predicted'
os.makedirs(output_dir, exist_ok=True)

# --- Load model ---
model = SimpleCNN().cpu()
model.load_state_dict(torch.load('denoising_cnn.pth', map_location='cpu'))
model.eval()

# --- Load data ---
dataset = NoisyCleanDataset(noisy_dir, clean_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# --- Evaluate ---
total_mse = 0.0
total_psnr = 0.0

print("Evaluating images...")
for i, (noisy, clean, filenames) in enumerate(dataloader):
    with torch.no_grad():
        output = model(noisy)
        mse = nn.functional.mse_loss(output, clean)
        psnr = calculate_psnr(output, clean)

        total_mse += mse.item()
        total_psnr += psnr.item()

        # Save denoised image
        out_img = output.squeeze(0).permute(1, 2, 0).numpy()
        out_img = (out_img * 255).clip(0, 255).astype('uint8')
        img_pil = Image.fromarray(out_img)
        img_pil.save(os.path.join(output_dir, filenames[0]))

        print(f"{i+1}/{len(dataloader)} - {filenames[0]} - PSNR: {psnr.item():.2f} dB")

# --- Final results ---
avg_mse = total_mse / len(dataloader)
avg_psnr = total_psnr / len(dataloader)

print("\nEvaluation complete.")
print(f"Average MSE  : {avg_mse:.6f}")
print(f"Average PSNR : {avg_psnr:.2f} dB")


