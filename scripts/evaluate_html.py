import os
from PIL import Image
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

def compare_images(original_path, enhanced_path):
    img1 = cv2.imread(original_path)
    img2 = cv2.imread(enhanced_path)

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    ssim_val = ssim(img1, img2, channel_axis=2)
    psnr_val = cv2.PSNR(img1, img2)
    return ssim_val, psnr_val

def generate_html_report(gt_folder, out_folder, html_path="evaluation_report.html"):
    rows = []
    for f in sorted(os.listdir(gt_folder)):
        gt_path = os.path.join(gt_folder, f)
        out_path = os.path.join(out_folder, f.replace('.png', '_out.png'))

        if not os.path.exists(gt_path) or not os.path.exists(out_path):
            continue

        ssim_val, psnr_val = compare_images(gt_path, out_path)
        rows.append(f"""
        <tr>
            <td><img src="{gt_path}" width="300"></td>
            <td><img src="{out_path}" width="300"></td>
            <td>{ssim_val:.4f}</td>
            <td>{psnr_val:.2f}</td>
        </tr>
        """)

    html = f"""
    <html><body>
    <h1>Real-ESRGAN Evaluation Report</h1>
    <table border="1" cellpadding="10">
    <tr><th>Ground Truth</th><th>Enhanced</th><th>SSIM</th><th>PSNR</th></tr>
    {''.join(rows)}
    </table>
    </body></html>
    """

    with open(html_path, "w") as f:
        f.write(html)

    print(f"✅ Report written to: {html_path}")

if __name__ == "__main__":
    generate_html_report("datasets/custom/val/hr", "results")
