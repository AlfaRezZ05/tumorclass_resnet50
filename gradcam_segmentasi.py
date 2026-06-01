import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import cv2
from torchvision import datasets, transforms, models
from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURASI
# ============================================================
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE  = 224
TEST_PATH = 'dataset/Testing'

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ============================================================
# FUNGSI SEGMENTASI OTAK (SKULL STRIPPING)
# ============================================================
def segment_brain(img_np):
    """
    Segmentasi otak dari citra MRI menggunakan OpenCV.
    Input : img_np → numpy array float32 [0,1] RGB (H x W x 3)
    Output: mask   → numpy array uint8 [0,255] (H x W)
    """
    # Convert ke grayscale
    img_uint8 = (img_np * 255).astype(np.uint8)
    gray      = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

    # Threshold Otsu untuk pisahkan otak dari background
    _, thresh = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations untuk bersihkan noise
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  kernel, iterations=2)

    # Ambil contour terbesar (= otak)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)

    # Fill holes dalam mask
    flood = mask.copy()
    h, w  = mask.shape
    flood_mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes  = cv2.bitwise_not(flood)
    mask   = mask | holes

    return mask

def apply_brain_mask(img_np, mask):
    """
    Terapkan mask otak ke gambar.
    Area luar otak dijadikan hitam.
    """
    mask_3ch = np.stack([mask/255.0]*3, axis=-1).astype(np.float32)
    return img_np * mask_3ch

def apply_mask_to_cam(cam_mask, brain_mask):
    """
    Terapkan brain mask ke heatmap CAM.
    Area luar otak dijadikan 0 (tidak aktif).
    """
    brain_float = brain_mask.astype(np.float32) / 255.0
    return cam_mask * brain_float

# ============================================================
# FUNGSI GENERATE GRAD-CAM + SEGMENTASI
# ============================================================
def generate_gradcam_segmented(model_path, train_path,
                                skenario_name, output_dir):
    print(f"\nGenerating Segmented Grad-CAM: {skenario_name}")
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    test_dataset  = datasets.ImageFolder(TEST_PATH,  transform=transform)
    class_names   = train_dataset.classes
    num_classes   = len(class_names)

    # Load model
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    target_layer = [model.layer4[-1]]
    gradcam      = GradCAM(model=model,  target_layers=target_layer)
    scorecam     = ScoreCAM(model=model, target_layers=target_layer)

    shown = {i: False for i in range(num_classes)}

    for img_tensor, label in test_dataset:
        if shown[label]:
            continue
        shown[label] = True

        inp     = img_tensor.unsqueeze(0).to(DEVICE)
        targets = [ClassifierOutputTarget(label)]

        # Grad-CAM & Score-CAM masks
        gc_mask = gradcam(input_tensor=inp,  targets=targets)
        sc_mask = scorecam(input_tensor=inp, targets=targets)

        # Denormalize gambar asli
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        orig = img_tensor.permute(1, 2, 0).numpy()
        orig = std * orig + mean
        orig = np.clip(orig, 0, 1).astype(np.float32)

        # ---- SEGMENTASI OTAK ----
        brain_mask     = segment_brain(orig)
        orig_segmented = apply_brain_mask(orig, brain_mask)

        # Terapkan brain mask ke CAM
        gc_masked = apply_mask_to_cam(gc_mask[0], brain_mask)
        sc_masked = apply_mask_to_cam(sc_mask[0], brain_mask)

        # Visualisasi CAM di atas gambar tersegmentasi
        gc_vis = show_cam_on_image(orig_segmented, gc_masked, use_rgb=True)
        sc_vis = show_cam_on_image(orig_segmented, sc_masked, use_rgb=True)

        # Hitung probabilitas prediksi
        with torch.no_grad():
            outputs   = model(inp)
            probs     = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        pred_label = np.argmax(probs)

        # ---- PLOT ----
        fig = plt.figure(figsize=(17, 5))
        gs  = gridspec.GridSpec(1, 4, width_ratios=[3, 3, 3, 2.5],
                                wspace=0.05)

        # Gambar 1: Original
        ax0 = fig.add_subplot(gs[0])
        ax0.imshow(orig)
        ax0.set_title('Original', fontsize=12, fontweight='bold')
        ax0.axis('off')

        # Gambar 2: Grad-CAM (tersegmentasi)
        ax1 = fig.add_subplot(gs[1])
        ax1.imshow(gc_vis)
        ax1.set_title(f'Grad-CAM\npred: {class_names[pred_label]}',
                      fontsize=11)
        ax1.axis('off')

        # Gambar 3: Score-CAM (tersegmentasi)
        ax2 = fig.add_subplot(gs[2])
        ax2.imshow(sc_vis)
        ax2.set_title(f'Score-CAM\npred: {class_names[pred_label]}',
                      fontsize=11)
        ax2.axis('off')

        # Panel 4: True label + persentase
        ax3 = fig.add_subplot(gs[3])
        ax3.axis('off')
        ax3.set_facecolor('#f8f8f8')

        ax3.text(0.05, 0.97,
                 f'True Label:',
                 transform=ax3.transAxes,
                 fontsize=10, color='#333333',
                 verticalalignment='top')
        ax3.text(0.05, 0.90,
                 f'{class_names[label]}',
                 transform=ax3.transAxes,
                 fontsize=11, fontweight='bold', color='#1a6bbd',
                 verticalalignment='top')

        ax3.text(0.05, 0.80,
                 'Predicted Label:',
                 transform=ax3.transAxes,
                 fontsize=10, fontweight='bold', color='#333333',
                 verticalalignment='top')

        # Persentase tiap kelas (urut dari tertinggi)
        sorted_idx = np.argsort(probs)[::-1]
        y_pos = 0.71
        for idx in sorted_idx:
            pct    = probs[idx] * 100
            color  = '#2ecc71' if idx == pred_label else '#555555'
            weight = 'bold'    if idx == pred_label else 'normal'
            ax3.text(0.05, y_pos,
                     f'{class_names[idx]}: {pct:.2f}%',
                     transform=ax3.transAxes,
                     fontsize=9.5, fontweight=weight, color=color,
                     verticalalignment='top')
            y_pos -= 0.115

        plt.suptitle(f'True Label: {class_names[label]} | '
                     f'Skenario: {skenario_name}',
                     fontsize=12, fontweight='bold', y=1.01)
        plt.tight_layout()

        save_path = os.path.join(output_dir,
                                  f'cam_{class_names[label]}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

        print(f"  ✓ {class_names[label]} → "
              f"pred: {class_names[pred_label]} "
              f"({probs[pred_label]*100:.2f}%)")

        if all(shown.values()):
            break

    print(f"\nSelesai! Tersimpan di: {output_dir}")

# ============================================================
# JALANKAN KEDUA SKENARIO
# ============================================================
generate_gradcam_segmented(
    model_path    = 'results/imbalanced/resnet50_model.pth',
    train_path    = 'dataset/Imbalanced',
    skenario_name = 'Imbalanced',
    output_dir    = 'results/imbalanced/gradcam_segmented'
)

generate_gradcam_segmented(
    model_path    = 'results/balanced_gan/resnet50_model.pth',
    train_path    = 'dataset/Balanced_GAN',
    skenario_name = 'Balanced GAN',
    output_dir    = 'results/balanced_gan/gradcam_segmented'
)

print("\n====== SEMUA SELESAI ======")
print("Output ada di:")
print("  - results/imbalanced/gradcam_segmented/")
print("  - results/balanced_gan/gradcam_segmented/")