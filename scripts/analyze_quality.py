import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import cv2

def compare_images(original_path, enhanced_path):
    orig = np.array(Image.open(original_path).convert("RGB"))
    enh = np.array(Image.open(enhanced_path).convert("RGB"))

    ssim_score = ssim(orig, enh, channel_axis=2)
    psnr_score = cv2.PSNR(orig, enh)

    print(f"SSIM: {ssim_score:.4f}")
    print(f"PSNR: {psnr_score:.2f} dB")

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(orig)
    ax[0].set_title("Original")
    ax[0].axis("off")
    ax[1].imshow(enh)
    ax[1].set_title("Enhanced")
    ax[1].axis("off")
    plt.tight_layout()
    plt.show()

    return ssim_score, psnr_score

# Example
# compare_images("test_images/download5.png", "results/download5_out.png")
