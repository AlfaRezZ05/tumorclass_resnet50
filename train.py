import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, precision_score,
                             recall_score, f1_score, accuracy_score)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURASI
# ============================================================
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 30
LR          = 0.001
VAL_SPLIT   = 0.2
TEST_PATH   = 'dataset/Testing'

print(f"Device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")

# ============================================================
# TRANSFORMS
# ============================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ============================================================
# FUNGSI TRAINING
# ============================================================
def train_model(train_path, skenario_name, output_dir):
    print(f"\n{'='*60}")
    print(f"TRAINING SKENARIO: {skenario_name}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    full_dataset = datasets.ImageFolder(train_path, transform=transform)
    test_dataset = datasets.ImageFolder(TEST_PATH,  transform=transform)
    class_names  = full_dataset.classes
    num_classes  = len(class_names)

    print(f"Kelas: {class_names}")
    print(f"Total train data: {len(full_dataset)}")
    print(f"Total test data : {len(test_dataset)}")

    # Split train/validation
    val_size   = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    print(f"Train: {train_size} | Validation: {val_size}")

    train_loader = DataLoader(train_ds,   batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,     batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --------------------------------------------------------
    # MODEL ResNet50
    # --------------------------------------------------------
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------
    history = {'train_acc': [], 'val_acc': [],
               'train_loss': [], 'val_loss': []}

    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item()
            _, preds       = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)

        # --- VALIDATION ---
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs   = model(imgs)
                loss      = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds   = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        t_acc  = train_correct / train_total
        v_acc  = val_correct   / val_total
        t_loss = train_loss    / len(train_loader)
        v_loss = val_loss      / len(val_loader)

        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)

        scheduler.step()
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] "
              f"Train Acc: {t_acc:.4f} | Val Acc: {v_acc:.4f} | "
              f"Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")

    # Simpan model
    torch.save(model.state_dict(), os.path.join(output_dir, 'resnet50_model.pth'))

    # --------------------------------------------------------
    # GRAFIK ACCURACY & LOSS
    # --------------------------------------------------------
    epochs_range = range(1, EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_range, history['train_acc'], 'b-o', label='Train Accuracy')
    ax1.plot(epochs_range, history['val_acc'],   'o-', label='Val Accuracy', color='orange')
    ax1.set_title(f'Accuracy - {skenario_name}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_range, history['train_loss'], 'g--o', label='Train Loss')
    ax2.plot(epochs_range, history['val_loss'],   'r--o', label='Val Loss')
    ax2.set_title(f'Loss - {skenario_name}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_loss.png'), dpi=150)
    plt.show()
    print(f"Grafik accuracy & loss tersimpan")

    # --------------------------------------------------------
    # EVALUASI DI TEST SET
    # --------------------------------------------------------
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs   = imgs.to(DEVICE)
            outputs = model(imgs)
            probs   = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Confusion Matrix - {skenario_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.show()
    print(f"Confusion matrix tersimpan")

    # --------------------------------------------------------
    # GRAFIK METRIK PER KELAS
    # --------------------------------------------------------
    precision  = precision_score(all_labels, all_preds, average=None)
    recall     = recall_score(all_labels, all_preds, average=None)
    f1         = f1_score(all_labels, all_preds, average=None)
    accuracy   = [accuracy_score(all_labels == i, all_preds == i)
                  for i in range(num_classes)]
    specificity = []
    for i in range(num_classes):
        tn = np.sum((all_labels != i) & (all_preds != i))
        fp = np.sum((all_labels != i) & (all_preds == i))
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

    x      = np.arange(num_classes)
    width  = 0.15
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 2*width, accuracy,    width, label='Accuracy',    color='steelblue')
    ax.bar(x - width,   precision,   width, label='Precision',   color='orange')
    ax.bar(x,           recall,      width, label='Recall',      color='green')
    ax.bar(x + width,   specificity, width, label='Specificity', color='red')
    ax.bar(x + 2*width, f1,          width, label='F1-Score',    color='purple')
    ax.set_xlabel('Kelas')
    ax.set_ylabel('Score')
    ax.set_title(f'Evaluation Metrics per Class - {skenario_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=15)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_metrics.png'), dpi=150)
    plt.show()
    print(f"Grafik metrik tersimpan")

    # --------------------------------------------------------
    # ROC / AUC
    # --------------------------------------------------------
    y_bin = label_binarize(all_labels, classes=range(num_classes))
    fig, ax = plt.subplots(figsize=(9, 7))
    colors  = ['blue','orange','green','red','purple','brown','pink','gray']
    auc_scores = []
    for i, cname in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc     = auc(fpr, tpr)
        auc_scores.append(roc_auc)
        ax.plot(fpr, tpr, color=colors[i % len(colors)],
                label=f'{cname} (AUC={roc_auc:.3f})')

    ax.plot([0,1],[0,1],'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {skenario_name}')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_auc.png'), dpi=150)
    plt.show()
    macro_auc = np.mean(auc_scores)
    print(f"Grafik ROC/AUC tersimpan | Macro AUC: {macro_auc:.4f}")

    # --------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------
    print("\nGenerating Grad-CAM visualizations...")
    try:
        from pytorch_grad_cam import GradCAM, ScoreCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        target_layer = [model.layer4[-1]]
        gradcam  = GradCAM(model=model, target_layers=target_layer)
        scorecam = ScoreCAM(model=model, target_layers=target_layer)

        test_full = datasets.ImageFolder(TEST_PATH, transform=transform)
        os.makedirs(os.path.join(output_dir, 'gradcam'), exist_ok=True)

        # Ambil 1 sample per kelas
        shown = {i: False for i in range(num_classes)}
        for img_tensor, label in test_full:
            if shown[label]:
                continue
            shown[label] = True

            inp = img_tensor.unsqueeze(0).to(DEVICE)
            targets = [ClassifierOutputTarget(label)]

            # Grad-CAM
            gc_mask  = gradcam(input_tensor=inp, targets=targets)
            # Score-CAM
            sc_mask  = scorecam(input_tensor=inp, targets=targets)

            # Denormalize gambar asli
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            orig = img_tensor.permute(1,2,0).numpy()
            orig = std * orig + mean
            orig = np.clip(orig, 0, 1).astype(np.float32)

            gc_vis  = show_cam_on_image(orig, gc_mask[0], use_rgb=True)
            sc_vis  = show_cam_on_image(orig, sc_mask[0], use_rgb=True)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(orig)
            axes[0].set_title('Original')
            axes[0].axis('off')
            axes[1].imshow(gc_vis)
            axes[1].set_title(f'Grad-CAM\n{class_names[label]}')
            axes[1].axis('off')
            axes[2].imshow(sc_vis)
            axes[2].set_title(f'Score-CAM\n{class_names[label]}')
            axes[2].axis('off')

            plt.suptitle(f'True Label: {class_names[label]}', fontsize=13)
            plt.tight_layout()
            save_path = os.path.join(output_dir, 'gradcam',
                                     f'cam_{class_names[label]}.png')
            plt.savefig(save_path, dpi=150)
            plt.show()

            if all(shown.values()):
                break

        print("Grad-CAM visualizations tersimpan di folder gradcam/")

    except ImportError:
        print("grad-cam belum terinstall, skip Grad-CAM")
        print("Install dengan: pip install grad-cam")

    # --------------------------------------------------------
    # RINGKASAN AKHIR
    # --------------------------------------------------------
    overall_acc = accuracy_score(all_labels, all_preds)
    print(f"\n{'='*60}")
    print(f"HASIL AKHIR - {skenario_name}")
    print(f"{'='*60}")
    print(f"Overall Accuracy : {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    print(f"Macro AUC        : {macro_auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=class_names))
    print(f"\nOutput tersimpan di folder: {output_dir}/")

    return model, history

# ============================================================
# JALANKAN KEDUA SKENARIO
# ============================================================
print("BRAIN TUMOR MRI CLASSIFICATION - ResNet50")
print("=" * 60)

# Skenario 1: Imbalanced
model_imb, history_imb = train_model(
    train_path     = 'dataset/Imbalanced',
    skenario_name  = 'Imbalanced',
    output_dir     = 'results/imbalanced'
)

# Skenario 2: Balanced
model_bal, history_bal = train_model(
    train_path     = 'dataset/Balanced',
    skenario_name  = 'Balanced',
    output_dir     = 'results/balanced'
)

print("\n" + "=" * 60)
print("SEMUA SKENARIO SELESAI!")
print("=" * 60)
print("Output ada di folder:")
print("  - results/imbalanced/")
print("  - results/balanced/")