# Name: HITESH
# Roll No: 2301010280
# Course: Image Processing
# Assignment: Smart Document Scanner


import cv2
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('Sample.jpg')

# Resize
img = cv2.resize(img, (512, 512))

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Show images
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale")

plt.show()



# Downsampling
img_256 = cv2.resize(gray, (256,256))
img_128 = cv2.resize(gray, (128,128))

# Upscale for viewing
img_256_up = cv2.resize(img_256, (512,512))
img_128_up = cv2.resize(img_128, (512,512))

# Display
plt.figure(figsize=(8,6))

plt.subplot(1,3,1)
plt.imshow(gray, cmap='gray')
plt.title("512x512")

plt.subplot(1,3,2)
plt.imshow(img_256_up, cmap='gray')
plt.title("256x256")

plt.subplot(1,3,3)
plt.imshow(img_128_up, cmap='gray')
plt.title("128x128")

plt.show()


import numpy as np

# Quantization function
def quantize(img, levels):
    return np.floor(img / (256/levels)) * (256/levels)

img_8bit = gray
img_4bit = quantize(gray, 16)
img_2bit = quantize(gray, 4)

# Display
plt.figure(figsize=(8,6))

plt.subplot(1,3,1)
plt.imshow(img_8bit, cmap='gray')
plt.title("8-bit (256 levels)")

plt.subplot(1,3,2)
plt.imshow(img_4bit, cmap='gray')
plt.title("4-bit (16 levels)")

plt.subplot(1,3,3)
plt.imshow(img_2bit, cmap='gray')
plt.title("2-bit (4 levels)")

plt.show()


cv2.imwrite("outputs/original.png", img)
cv2.imwrite("outputs/gray.png", gray)
cv2.imwrite("outputs/256.png", img_256_up)
cv2.imwrite("outputs/128.png", img_128_up)
cv2.imwrite("outputs/4bit.png", img_4bit)
cv2.imwrite("outputs/2bit.png", img_2bit)


plt.figure(figsize=(10,8))

images = [
    img, gray,
    gray, img_256_up, img_128_up,
    gray, img_4bit, img_2bit
]

titles = [
    "Original", "Grayscale",
    "512x512", "256x256", "128x128",
    "8-bit", "4-bit", "2-bit"
]

for i in range(8):
    plt.subplot(3,3,i+1)
    if i == 0:
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
