import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# Define the same SimpleCNN model as before
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
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

# Dataset class to load images
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, image_name)
        
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, image_name  # Return image and its filename

# Define transformations (resize to 256x256 and convert to tensor)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load test dataset
test_dataset = ImageDataset(root_dir='dataset', transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Initialize the model
model = SimpleCNN()
model.load_state_dict(torch.load('simple_cnn_model.pth'))
model.eval()  # Set the model to evaluation mode

# Create the 'predicted' folder if it doesn't exist
os.makedirs('predicted', exist_ok=True)

# Evaluate and save output images
with torch.no_grad():  # No need to compute gradients during evaluation
    for idx, (inputs, filenames) in enumerate(test_dataloader):
        outputs = model(inputs)
        
        # Convert the output tensor to an image and save it
        for i in range(len(filenames)):
            output_img = outputs[i].cpu().clamp(0, 1)  # Clamp values to [0, 1]
            output_img = transforms.ToPILImage()(output_img)  # Convert to PIL image
            
            # Save the output image with the same name as input image but in the 'predicted' folder
            output_path = os.path.join('predicted', filenames[i])
            output_img.save(output_path)

print("Evaluation complete and images saved in 'predicted' folder.")

