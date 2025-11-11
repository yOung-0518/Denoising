#%%
from PIL import Image
import os
import numpy as np
import cv2
import glob
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import pandas as pd

clean_root = "D:/DIP/BSDS300/clean"
noise_root = "D:/DIP/BSDS300/Gaussian_Noise"

image_paths = glob.glob(os.path.join(clean_root, "**", "*.jpg"), recursive=True)

#%%1. Gaussian noise
def GaussianNoise(image, mean=0, std=25):
    image_np = np.array(image).astype(np.float32)
    noise = np.random.normal(mean, std, image_np.shape)
    n_image = np.clip(image_np + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(n_image)

for path in image_paths:

    img = Image.open(path).convert("L")
    noisy_img = GaussianNoise(img)

    rel_path = os.path.relpath(path, clean_root)  
    save_path = os.path.join(noise_root, rel_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
   
    noisy_img.save(save_path)


#%%
# 2. Gaussian Filter
def gaussian_filter(image, kernel_size, sigma=1):
    image_np = np.array(image)
    blurred = cv2.GaussianBlur(image_np, ksize=kernel_size, sigmaX=sigma)
    return Image.fromarray(blurred)

input_dir = "D:/DIP/BSDS300/Gaussian_Noise/test"
image_paths = glob.glob(os.path.join(input_dir, "**", "*.jpg"), recursive=True)
gn_images = [Image.open(p).convert("L") for p in image_paths]

dn_3 = [gaussian_filter(img, (3, 3)) for img in gn_images]
dn_11 = [gaussian_filter(img, (11, 11)) for img in gn_images]

output_root_3x3 = "D:/DIP/Denoising/Gaussian_Filter/3x3"
output_root_11x11 = "D:/DIP/Denoising/Gaussian_Filter/11x11"

def save_filtered_images(image_paths, filtered_images, output_root):
    for path, img in zip(image_paths, filtered_images):
        rel_path = os.path.relpath(path, input_dir)  
        save_path = os.path.join(output_root, rel_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path)

save_filtered_images(image_paths, dn_3, output_root_3x3)
save_filtered_images(image_paths, dn_11, output_root_11x11)

#%%
# 3. Median Filter
def median_filter(image, kernel_size):
    np_img = np.array(image.convert("L"))  # 흑백으로 강제 변환
    filtered = cv2.medianBlur(np_img, kernel_size)
    return Image.fromarray(filtered)

input_dir = "D:/DIP/BSDS300/Gaussian_Noise/test"
image_paths = glob.glob(os.path.join(input_dir, "**", "*.jpg"), recursive=True)
gn_images = [Image.open(p).convert("L") for p in image_paths]

output_root = "D:/DIP/Denoising/Median_Filter"

def process_and_save_median(images, image_paths, kernel_size):
    output_subdir = f"{kernel_size}x{kernel_size}"
    for path, img in zip(image_paths, images):
        filtered = median_filter(img, kernel_size=kernel_size)
        rel_path = os.path.relpath(path, input_dir)
        save_path = os.path.join(output_root, output_subdir, rel_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        filtered.save(save_path)

process_and_save_median(gn_images, image_paths, kernel_size=3)
process_and_save_median(gn_images, image_paths, kernel_size=11)

print("Median filtering 3x3 & 11x11 done.")

# %%
clean_dir = "D:/DIP/BSDS300/clean/test"

filter_dirs = {
    "Gaussian 3x3": "D:/DIP/Denoising/Gaussian_Filter/3x3",
    "Gaussian 11x11": "D:/DIP/Denoising/Gaussian_Filter/11x11",
    "Median 3": "D:/DIP/Denoising/Median_Filter/3",
    "Median 11": "D:/DIP/Denoising/Median_Filter/11",
}

def evaluate_filter_results(clean_dir, result_dir):
    psnr_list, ssim_list = [], []
    for filename in os.listdir(clean_dir):
        clean_path = os.path.join(clean_dir, filename)
        result_path = os.path.join(result_dir, filename)
        
        if not os.path.exists(result_path):
            print(f"{filename} 없음 — {result_path}")
            continue
        
        clean_img = Image.open(clean_path).convert("L")
        result_img = Image.open(result_path).convert("L")
        
        clean_np = np.array(clean_img, dtype=np.float32) / 255.0
        result_np = np.array(result_img, dtype=np.float32) / 255.0
        
        psnr_val = psnr(clean_np, result_np, data_range=1.0)
        ssim_val = ssim(clean_np, result_np, data_range=1.0)
        
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
    
    return np.mean(psnr_list), np.mean(ssim_list)


results = []
for name, path in filter_dirs.items():
    avg_psnr, avg_ssim = evaluate_filter_results(clean_dir, path)
    results.append({
        "Filter": name,
        "Average PSNR (dB)": round(avg_psnr, 2),
        "Average SSIM": round(avg_ssim, 4)
    })

df = pd.DataFrame(results)
print(df.to_markdown(index=False))

# %%
