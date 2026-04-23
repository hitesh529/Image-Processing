# Name: HITESH
# Roll No: 2301010280
# Course: Image Processing
# Assignment: Image Restoration


import cv2
import numpy as np
import matplotlib.pyplot as plt

#Load image
img = cv2.imread('Sample.jpg')
img = cv2.resize(img, (512, 512))

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#SHOW ORIGINAL + GRAY
plt.figure(figsize=(6,4))

plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale")
plt.axis('off')

plt.tight_layout()
plt.show()

#ADD NOISE

# Gaussian Noise
gaussian = gray + np.random.normal(0, 25, gray.shape)
gaussian = np.clip(gaussian, 0, 255).astype(np.uint8)

# Salt & Pepper Noise
sp = gray.copy()
prob = 0.02

salt = np.random.rand(*gray.shape) < prob
pepper = np.random.rand(*gray.shape) < prob

sp[salt] = 255
sp[pepper] = 0

#FILTERS

# Mean Filter
mean_g = cv2.blur(gaussian, (5,5))
mean_sp = cv2.blur(sp, (5,5))

# Median Filter
median_g = cv2.medianBlur(gaussian, 5)
median_sp = cv2.medianBlur(sp, 5)

# Gaussian Filter
gauss_g = cv2.GaussianBlur(gaussian, (5,5), 0)
gauss_sp = cv2.GaussianBlur(sp, (5,5), 0)

#MSE & PSNR

def mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def psnr(img1, img2):
    m = mse(img1, img2)
    if m == 0:
        return 100
    return 20 * np.log10(255 / np.sqrt(m))

print("\n--- Performance ---")

print("Gaussian + Mean:", mse(gray, mean_g), psnr(gray, mean_g))
print("Gaussian + Median:", mse(gray, median_g), psnr(gray, median_g))
print("Gaussian + Gaussian:", mse(gray, gauss_g), psnr(gray, gauss_g))

print("SaltPepper + Mean:", mse(gray, mean_sp), psnr(gray, mean_sp))
print("SaltPepper + Median:", mse(gray, median_sp), psnr(gray, median_sp))
print("SaltPepper + Gaussian:", mse(gray, gauss_sp), psnr(gray, gauss_sp))

#FINAL COMPARISON FIGURE

plt.figure(figsize=(10,8))

images = [
    gray, gaussian, sp,
    mean_g, median_g, gauss_g,
    mean_sp, median_sp, gauss_sp
]

titles = [
    "Original Image", "Gaussian Noise", "Salt & Pepper",
    "Mean (G)", "Median (G)", "Gaussian (G)",
    "Mean (SP)", "Median (SP)", "Gaussian (SP)"
]

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
