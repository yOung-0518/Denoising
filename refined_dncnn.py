#%%
import os, glob, random, shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class DnCNN1(nn.Module):
    def __init__(self, channels=1, num_of_layers=5, kernel_size=3):
        super(DnCNN1, self).__init__()
        padding = kernel_size // 2
        features = 32
        layers = []

        layers.append(nn.Conv2d(channels, features, kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(features, features, kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(features, channels, kernel_size, padding=padding, bias=False))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.dncnn(x)


def is_vertical(img: Image.Image):
    return img.height > img.width

def rotate_if_vertical(img: Image.Image) -> Image.Image:
    return img.transpose(Image.ROTATE_90) if is_vertical(img) else img

class DenoiseDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir):
        self.noisy_paths = sorted(glob.glob(os.path.join(noisy_dir, "*.*")))
        self.clean_paths = []
        for path in self.noisy_paths:
            filename = os.path.basename(path)
            clean_path = os.path.join(clean_dir, filename)
            if os.path.exists(clean_path):
                self.clean_paths.append(clean_path)
        self.noisy_paths = self.noisy_paths[:len(self.clean_paths)]

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):

        noisy = Image.open(self.noisy_paths[idx]).convert("L")
        clean = Image.open(self.clean_paths[idx]).convert("L")

        noisy = rotate_if_vertical(noisy)
        clean = rotate_if_vertical(clean)

        noisy_np = np.expand_dims(np.array(noisy), axis=0)  
        clean_np = np.expand_dims(np.array(clean), axis=0)

        noisy = torch.tensor(noisy_np, dtype=torch.float32) / 255.
        clean = torch.tensor(clean_np, dtype=torch.float32) / 255.

        return noisy, clean


clean_train = "D:/DIP/BSDS300/clean/train"
clean_val   = "D:/DIP/BSDS300/clean/val"
noise_train = "D:/DIP/BSDS300/Gaussian_Noise/train"
noise_val   = "D:/DIP/BSDS300/Gaussian_Noise/val"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

save_path = "D:/DIP/DnCNN-PyTorch-master/logs/DnCNN-finetuned/net_finetuned_color.pth"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

model = DnCNN1(channels=1, num_of_layers=5, kernel_size=3)
model = nn.DataParallel(model).to(device)
model.train()

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

train_loader = DataLoader(DenoiseDataset(noise_train, clean_train), batch_size=4, shuffle=True)
val_loader   = DataLoader(DenoiseDataset(noise_val, clean_val), batch_size=1, shuffle=False)

num_epochs = 10
train_losses = []

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0
    for noisy, clean in train_loader:
        noisy, clean = noisy.to(device), clean.to(device)
        output = model(noisy)
        loss = criterion(output, clean)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss) 

    val_psnrs = []
    
    model.eval()
    total_psnr = 0
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            psnr_val = psnr(clean.squeeze().cpu().numpy(), output.squeeze().cpu().numpy(), data_range=1.0)
            total_psnr += psnr_val
    avg_psnr = total_psnr / len(val_loader)
    val_psnrs.append(avg_psnr)  
    print(f"Epoch {epoch} | Train Loss: {total_loss/len(train_loader):.6f} | Val PSNR: {avg_psnr:.2f} dB")

torch.save(model.state_dict(), save_path)
print("학습 완료. 모델 저장됨:", save_path)

#%%

test_noisy_dir = "D:/DIP/BSDS300/Gaussian_Noise/test"
test_clean_dir = "D:/DIP/BSDS300/clean/test"
output_dir = "D:/DIP/denoising/DnCNN/denoised(5, 10, 32, 3)"
os.makedirs(output_dir, exist_ok=True)

test_loader = DataLoader(DenoiseDataset(test_noisy_dir, test_clean_dir), batch_size=1, shuffle=False)

model.eval()
total_psnr = 0
total_ssim = 0
total_mse = 0
total_acc = 0

with torch.no_grad():
    for noisy_path, (noisy, clean) in zip(test_loader.dataset.noisy_paths, test_loader):
        noisy, clean = noisy.to(device), clean.to(device)
        output = model(noisy)

        output_np = output.squeeze().cpu().numpy()
        clean_np = clean.squeeze().cpu().numpy()

        psnr_val = psnr(clean_np, output_np, data_range=1.0)
        ssim_val = ssim(clean_np, output_np, data_range=1.0)

        total_psnr += psnr_val
        total_ssim += ssim_val
        total_mse += F.mse_loss(output, clean).item()

        filename = os.path.basename(noisy_path)
        out_img = (output_np * 255).round().clip(0, 255).astype(np.uint8)
        Image.fromarray(out_img).save(os.path.join(output_dir, filename))


n = len(test_loader)
print(f"Test 결과과:")
print(f" - 평균 PSNR     : {total_psnr/n:.2f} dB")
print(f" - 평균 SSIM     : {total_ssim/n:.4f}")
print(f" - 평균 MSE      : {total_mse/n:.6f}")



# %%
