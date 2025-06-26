import matplotlib.pyplot as plt
import cv2
import os

def show_change(image1_path, image2_path):
    img1 = cv2.imread(image1_path, 0)
    img2 = cv2.imread(image2_path, 0)
    diff = cv2.absdiff(img1, img2)
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(img1, cmap='gray')
    plt.title("Before")
    plt.subplot(1,3,2)
    plt.imshow(img2, cmap='gray')
    plt.title("After")
    plt.subplot(1,3,3)
    plt.imshow(diff, cmap='hot')
    plt.title("Change Detected")
    plt.show()
