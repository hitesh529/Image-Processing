# Name: HITESH
# Roll No: 2301010280
# Course: Image Processing
# Assignment: Medical Image Compression & Segmentation


import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------- STEP 1: LOAD IMAGE ----------------
img = cv2.imread('Sample1.jpg', 0)   # grayscale directly
img = cv2.resize(img, (512, 512))

# ---------------- STEP 2: RLE COMPRESSION ----------------
def rle_encode(image):
    pixels = image.flatten()
    encoded = []
    
    prev = pixels[0]
    count = 1
    
    for pixel in pixels[1:]:
        if pixel == prev:
            count += 1
        else:
            encoded.append((prev, count))
            prev = pixel
            count = 1
    
    encoded.append((prev, count))
    return encoded

encoded = rle_encode(img)

original_size = img.size
compressed_size = len(encoded) * 2   # value + count

compression_ratio = original_size / compressed_size
savings = (1 - (compressed_size / original_size)) * 100

print("Compression Ratio:", compression_ratio)
print("Storage Savings (%):", savings)

# ---------------- STEP 3: SEGMENTATION ----------------
blur = cv2.GaussianBlur(img, (5,5), 0)
# Global Threshold
_, global_thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)

# Otsu Threshold
_, otsu_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# ---------------- STEP 4: MORPHOLOGICAL OPERATIONS ----------------

kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(otsu_thresh, cv2.MORPH_CLOSE, kernel)

# Dilation
dilated = cv2.dilate(otsu_thresh, kernel, iterations=1)

# Erosion
eroded = cv2.erode(otsu_thresh, kernel, iterations=1)

# ---------------- STEP 5: DISPLAY ALL ----------------

plt.figure(figsize=(10,8))

images = [img, global_thresh, otsu_thresh, opening, closing]
titles = ["Original", "Global", "Otsu", "Opening", "Closing"]

for i in range(len(images)):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
