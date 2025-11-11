
#%%
import os, glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models import DnCNN
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import random
import shutil

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

        noisy_np = np.array(noisy, dtype=np.float32) / 255.0  # [H, W]
        clean_np = np.array(clean, dtype=np.float32) / 255.0

        noisy_tensor = torch.tensor(noisy_np).unsqueeze(0)
        clean_tensor = torch.tensor(clean_np).unsqueeze(0)

        return noisy_tensor, clean_tensor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model_path = "D:/DIP/DnCNN-PyTorch-master/logs/DnCNN-S-25/net.pth"  # 사전학습된 가중치
save_path = "D:/DIP/DnCNN-PyTorch-master/logs/DnCNN-finetuned/net_finetuned.pth"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

model = DnCNN(channels=1).to(device)
model = nn.DataParallel(model).to(device)  
model.load_state_dict(torch.load(model_path, map_location=device))
model.train()

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()


clean_train = "D:/DIP/BSDS300/clean/train"
clean_val   = "D:/DIP/BSDS300/clean/val"
noise_train = "D:/DIP/BSDS300/Gaussian_Noise/train"
noise_val   = "D:/DIP/BSDS300/Gaussian_Noise/val"

train_loader = DataLoader(DenoiseDataset("D:/DIP/BSDS300/Gaussian_Noise/train", "D:/DIP/BSDS300/clean/train"), batch_size=4, shuffle=True)
val_loader   = DataLoader(DenoiseDataset("D:/DIP/BSDS300/Gaussian_Noise/val", "D:/DIP/BSDS300/clean/val"), batch_size=1, shuffle=False)

num_epochs = 10
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

    model.eval()
    total_psnr = 0
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            psnr_val = psnr(clean.squeeze().cpu().numpy(), output.squeeze().cpu().numpy(), data_range=1.0)
            total_psnr += psnr_val
    avg_psnr = total_psnr / len(val_loader)
    print(f"Epoch {epoch} | Train Loss: {total_loss/len(train_loader):.6f} | Val PSNR: {avg_psnr:.2f} dB")

torch.save(model.state_dict(), save_path)
print("Finetuning 완료. 모델 저장됨:", save_path)
#%%
import torchvision.transforms.functional as TF
import torch.nn.functional as F

output_dir = "D:/DIP/BSDS300/result/denoised(17, 10)"
os.makedirs(output_dir, exist_ok=True)

test_loader = DataLoader(DenoiseDataset("D:/DIP/BSDS300/Gaussian_Noise/test", "D:/DIP/BSDS300/clean/test"), batch_size=1, shuffle=False)
model.eval()
total_psnr = 0
total_mse = 0
total_ssim = 0

with torch.no_grad():
    for noisy_path, (noisy, clean) in zip(test_loader.dataset.noisy_paths, test_loader):
        noisy, clean = noisy.to(device), clean.to(device)
        output = model(noisy)

        output_np = output.squeeze().cpu().numpy()
        clean_np = clean.squeeze().cpu().numpy()

        filename = os.path.basename(noisy_path)
        save_path = os.path.join(output_dir, filename)

        out_img = (output_np * 255).round().clip(0, 255).astype(np.uint8)
        Image.fromarray(out_img).save(save_path)

        total_psnr += psnr(clean_np, output_np, data_range=1.0)
        total_ssim += ssim(clean_np, output_np, data_range=1.0)
        total_mse += F.mse_loss(output, clean).item()

n = len(test_loader)

print(f"Test 결과 :")
print(f" - 평균 PSNR     : {total_psnr/n:.2f} dB")
print(f" - 평균 SSIM     : {total_ssim/n:.4f}")
print(f" - 평균 MSE      : {total_mse/n:.6f}")

# %%
