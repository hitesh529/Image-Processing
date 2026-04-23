# Name: HITESH
# Roll No: 2301010280
# Course: Image Processing
# Assignment: Intelligent Image Processing System



import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

print("Welcome to Intelligent Image Processing System")

# ---------------- STEP 1: LOAD IMAGE ----------------
img = cv2.imread('Sample.jpg')
img = cv2.resize(img, (512, 512))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---------------- STEP 2: NOISE ----------------
gaussian = gray + np.random.normal(0, 25, gray.shape)
gaussian = np.clip(gaussian, 0, 255).astype(np.uint8)

sp = gray.copy()
prob = 0.02
salt = np.random.rand(*gray.shape) < prob
pepper = np.random.rand(*gray.shape) < prob
sp[salt] = 255
sp[pepper] = 0

# ---------------- STEP 3: RESTORATION ----------------
mean = cv2.blur(gaussian, (5,5))
median = cv2.medianBlur(sp, 5)
gauss = cv2.GaussianBlur(gaussian, (5,5), 0)

# Contrast Enhancement
enhanced = cv2.equalizeHist(gray)

# ---------------- STEP 4: SEGMENTATION ----------------
blur = cv2.GaussianBlur(enhanced, (5,5), 0)

_, otsu = cv2.threshold(blur, 0, 255,
                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = np.ones((5,5), np.uint8)

opening = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

# ---------------- STEP 5: EDGE + FEATURES ----------------
canny = cv2.Canny(gray, 100, 200)

contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_img = img.copy()

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 3000:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(contour_img, (x,y), (x+w, y+h), (0,255,0), 2)

orb = cv2.ORB_create()
kp, des = orb.detectAndCompute(gray, None)
orb_img = cv2.drawKeypoints(img, kp, None, color=(0,255,0))

# ---------------- STEP 6: METRICS ----------------
def mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def psnr(img1, img2):
    m = mse(img1, img2)
    if m == 0:
        return 100
    return 20 * np.log10(255 / np.sqrt(m))

print("\n--- Performance ---")
print("MSE:", mse(gray, enhanced))
print("PSNR:", psnr(gray, enhanced))
print("SSIM:", ssim(gray, enhanced))

# ---------------- STEP 7: FINAL DISPLAY ----------------
plt.figure(figsize=(12,10))

images = [
    img, gaussian, median,
    enhanced, otsu, contour_img,
    orb_img
]

titles = [
    "Original", "Noisy", "Restored",
    "Enhanced", "Segmented", "Contours",
    "ORB Features"
]

for i in range(len(images)):
    plt.subplot(3,3,i+1)
    
    if len(images[i].shape) == 3:
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(images[i], cmap='gray')
    
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()

print("\nConclusion:")
print("The system successfully enhances, restores, segments, and extracts features from the image.")
