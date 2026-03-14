import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

# --- 1. FUNGSI PENAMBAH NOISE (Poin 1 Lengkap) ---
def add_noise(img, noise_type):
    row, col = img.shape
    img_normalized = img / 255.0 # Normalisasi ke 0-1 untuk perhitungan noise
    
    if noise_type == "gaussian":
        gauss = np.random.normal(0, 0.1, (row, col))
        noisy = np.clip(img_normalized + gauss, 0, 1)
        return (noisy * 255).astype(np.uint8)
    
    elif noise_type == "sp":
        out = np.copy(img)
        prob = 0.05
        out[np.random.rand(row, col) < prob/2] = 255
        out[np.random.rand(row, col) < prob/2] = 0
        return out
    
    elif noise_type == "speckle":
        # Speckle: J = I + n*I (n adalah noise berdistribusi seragam/gauss)
        noise = np.random.randn(row, col) * 0.15 
        noisy = np.clip(img_normalized + img_normalized * noise, 0, 1)
        return (noisy * 255).astype(np.uint8)

# --- 2. FUNGSI EVALUASI (Poin 4) ---
def evaluate_filter(original, filtered, start_time, filter_name, noise_name):
    duration = time.time() - start_time
    m_val = mse(original, filtered)
    p_val = psnr(original, filtered, data_range=255)
    s_val = ssim(original, filtered, data_range=255)
    
    print(f"[{filter_name} on {noise_name}]")
    print(f"MSE: {m_val:.2f} | PSNR: {p_val:.2f}dB | SSIM: {s_val:.4f} | Waktu: {duration:.5f}s")
    print("-" * 60)
    return {"img": filtered, "name": filter_name, "psnr": p_val}

# --- MAIN PROGRAM ---

# Load Citra (Pastikan file ada atau gunakan citra default)
img_orig = cv2.imread('CitraAsli.jpeg', cv2.IMREAD_GRAYSCALE)
if img_orig is None:
    img_orig = (np.indices((256, 256)).sum(axis=0) % 255).astype(np.uint8) # Citra gradien dummy

# Generate 3 Variasi Korupsi
img_gauss = add_noise(img_orig, "gaussian")
img_sp = add_noise(img_orig, "sp")
img_speckle = add_noise(img_orig, "speckle")

results = []

# --- IMPLEMENTASI SEMUA FILTER ---

# A. Mean Filter (Linear - Kernel 3x3 & 7x7) pada Gaussian Noise
t = time.time()
results.append(evaluate_filter(img_orig, cv2.blur(img_gauss, (3,3)), t, "Mean 3x3", "Gaussian"))
t = time.time()
results.append(evaluate_filter(img_orig, cv2.blur(img_gauss, (7,7)), t, "Mean 7x7", "Gaussian"))

# B. Gaussian Filter (Linear - Sigma 1.0 & 2.5) pada Speckle Noise
t = time.time()
results.append(evaluate_filter(img_orig, cv2.GaussianBlur(img_speckle, (5,5), 1.0), t, "Gauss S1.0", "Speckle"))
t = time.time()
results.append(evaluate_filter(img_orig, cv2.GaussianBlur(img_speckle, (5,5), 2.5), t, "Gauss S2.5", "Speckle"))

# C. Median Filter (Non-Linear - Kernel 3x3 & 5x5) pada Salt & Pepper
t = time.time()
results.append(evaluate_filter(img_orig, cv2.medianBlur(img_sp, 3), t, "Median 3x3", "S&P"))
t = time.time()
results.append(evaluate_filter(img_orig, cv2.medianBlur(img_sp, 5), t, "Median 5x5", "S&P"))

# D. Max Filter (Non-Linear - Dilasi) pada Salt & Pepper
# Max filter akan menghilangkan 'pepper' (titik hitam) dengan memperluas area terang
t = time.time()
kernel_max = np.ones((3,3), np.uint8)
results.append(evaluate_filter(img_orig, cv2.dilate(img_sp, kernel_max), t, "Max Filter", "S&P"))

# --- 3. VISUAL INSPECTION (Poin 4) ---
plt.figure(figsize=(18, 10))

# Baris 1: Citra Asli dan 3 Jenis Noise
titles_noise = ["Original", "Gaussian Noise", "Salt & Pepper", "Speckle Noise"]
imgs_noise = [img_orig, img_gauss, img_sp, img_speckle]
for i in range(4):
    plt.subplot(2, 4, i+1)
    plt.imshow(imgs_noise[i], cmap='gray')
    plt.title(titles_noise[i])
    plt.axis('off')

# Baris 2: Sampel Hasil Filter untuk Perbandingan Sharpness
# Kita ambil beberapa contoh dari list results
sample_indices = [0, 2, 4, 6] # Mean, Gauss, Median, Max
for i, idx in enumerate(sample_indices):
    plt.subplot(2, 4, i+5)
    plt.imshow(results[idx]['img'], cmap='gray')
    plt.title(f"{results[idx]['name']}\nPSNR: {results[idx]['psnr']:.2f}")
    plt.axis('off')

plt.tight_layout()
plt.show()