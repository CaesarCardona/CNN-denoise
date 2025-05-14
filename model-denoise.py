import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# CNN for denoising
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

# Dataset: pairs noisy-clean
class NoisyCleanDataset(Dataset):
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

# Transform: resize + tensor
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Folders
noisy_folder = 'dataset-with-noise'
clean_folder = 'dataset'
output_folder = 'predicted'
os.makedirs(output_folder, exist_ok=True)

# Load data
dataset = NoisyCleanDataset(noisy_folder, clean_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Model
model = SimpleCNN().cpu()

# Loss + Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for noisy, clean, _ in dataloader:
        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - MSE Loss: {avg_loss:.6f}")

# Save model
torch.save(model.state_dict(), 'denoising_cnn.pth')
print("Training complete and model saved.")

# Predict and save denoised images
model.eval()
predict_loader = DataLoader(dataset, batch_size=1, shuffle=False)

with torch.no_grad():
    for noisy, _, filenames in predict_loader:
        output = model(noisy)
        output_img = output.squeeze(0).permute(1, 2, 0).numpy()
        output_img = (output_img * 255).clip(0, 255).astype('uint8')
        img_pil = Image.fromarray(output_img)
        img_pil.save(os.path.join(output_folder, filenames[0]))

print("Denoised images saved to 'predicted/'")

