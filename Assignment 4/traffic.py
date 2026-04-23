# Name: HITESH
# Roll No: 2301010280
# Course: Image Processing
# Assignment: Feature-Based Traffic Monitoring System


import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------- STEP 1: LOAD IMAGE ----------------
img = cv2.imread('Sample.jpg')
img = cv2.resize(img, (512, 512))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---------------- STEP 2: EDGE DETECTION ----------------

# Sobel
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobelx, sobely)
sobel = np.uint8(sobel)

# Canny
canny = cv2.Canny(gray, 100, 200)

# ---------------- STEP 3: CONTOURS ----------------
contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_img = img.copy()

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    
    # draw bounding box
    cv2.rectangle(contour_img, (x,y), (x+w, y+h), (0,255,0), 2)
    
    # print area & perimeter
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    print("Area:", area, "Perimeter:", perimeter)

# ---------------- STEP 4: FEATURE EXTRACTION (ORB) ----------------
orb = cv2.ORB_create()

keypoints, descriptors = orb.detectAndCompute(gray, None)

orb_img = cv2.drawKeypoints(img, keypoints, None, color=(0,255,0))

# ---------------- STEP 5: DISPLAY ----------------
plt.figure(figsize=(10,8))

images = [
    gray, sobel, canny,
    contour_img, orb_img
]

titles = [
    "Grayscale",
    "Sobel Edge",
    "Canny Edge",
    "Contours + Bounding Box",
    "ORB Keypoints"
]

for i in range(len(images)):
    plt.subplot(2,3,i+1)
    
    if i == 0 or i == 3 or i == 4:
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(images[i], cmap='gray')
    
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
