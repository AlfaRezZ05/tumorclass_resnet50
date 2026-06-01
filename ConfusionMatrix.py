import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURASI
# ============================================================
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 15
LR         = 0.001
N_FOLDS    = 5

print(f"Device: {DEVICE}")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ============================================================
# FUNGSI CV CONFUSION MATRIX
# ============================================================
def run_cv_confusion_matrix(data_path, skenario_name, output_dir):
    print(f"\n{'='*60}")
    print(f"CV CONFUSION MATRIX: {skenario_name}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    full_dataset = datasets.ImageFolder(data_path, transform=transform)
    class_names  = full_dataset.classes
    num_classes  = len(class_names)
    n_samples    = len(full_dataset)

    print(f"Kelas: {class_names}")
    print(f"Total data: {n_samples}")

    kfold   = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    indices = np.arange(n_samples)

    all_fold_cms    = []
    all_fold_preds  = []
    all_fold_labels = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")

        train_subset = Subset(full_dataset, train_idx)
        val_subset   = Subset(full_dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE,
                                  shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=0)

        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model    = model.to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        best_val_acc = 0

        for epoch in range(EPOCHS):
            # Train
            model.train()
            t_correct, t_total = 0, 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss    = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                _, preds   = torch.max(outputs, 1)
                t_correct += (preds == labels).sum().item()
                t_total   += labels.size(0)

            # Validation
            model.eval()
            v_correct, v_total = 0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    outputs  = model(imgs)
                    _, preds = torch.max(outputs, 1)
                    v_correct += (preds == labels).sum().item()
                    v_total   += labels.size(0)

            v_acc = v_correct / v_total
            if v_acc > best_val_acc:
                best_val_acc = v_acc
                torch.save(model.state_dict(),
                           os.path.join(output_dir, f'best_fold{fold+1}.pth'))

            scheduler.step()
            print(f"  Epoch [{epoch+1:02d}/{EPOCHS}] "
                  f"Train: {t_correct/t_total:.4f} | Val: {v_acc:.4f}")

        # Load best model
        model.load_state_dict(torch.load(
            os.path.join(output_dir, f'best_fold{fold+1}.pth'),
            map_location=DEVICE))
        model.eval()

        fold_preds, fold_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs    = imgs.to(DEVICE)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                fold_preds.extend(preds.cpu().numpy())
                fold_labels.extend(labels.numpy())

        fold_preds  = np.array(fold_preds)
        fold_labels = np.array(fold_labels)

        cm  = confusion_matrix(fold_labels, fold_preds)
        acc = accuracy_score(fold_labels, fold_preds)
        all_fold_cms.append(cm)
        all_fold_preds.extend(fold_preds)
        all_fold_labels.extend(fold_labels)

        print(f"  Fold {fold+1} Accuracy: {acc:.4f}")

        # Plot per fold
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names, ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'Confusion Matrix — {skenario_name} | Fold {fold+1} (Acc: {acc:.4f})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'cm_fold{fold+1}.png'), dpi=150)
        plt.show()
        plt.close()

    # ============================================================
    # SEMUA FOLD DALAM 1 GAMBAR
    # ============================================================
    fig, axes = plt.subplots(1, N_FOLDS, figsize=(5*N_FOLDS, 5))
    for fold, (cm, ax) in enumerate(zip(all_fold_cms, axes)):
        acc = np.trace(cm) / np.sum(cm)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[c[:5] for c in class_names],
                    yticklabels=[c[:5] for c in class_names],
                    ax=ax, cbar=False)
        ax.set_title(f'Fold {fold+1}\nAcc: {acc:.4f}', fontsize=11)
        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('True', fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', rotation=0,  labelsize=8)

    plt.suptitle(f'Confusion Matrix per Fold — {skenario_name}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cm_all_folds.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

    # ============================================================
    # CONFUSION MATRIX AGREGAT
    # ============================================================
    all_preds  = np.array(all_fold_preds)
    all_labels = np.array(all_fold_labels)
    cm_total   = confusion_matrix(all_labels, all_preds)
    acc_total  = accuracy_score(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_total, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Confusion Matrix Agregat (5-Fold) — {skenario_name}\n'
                 f'Overall Accuracy: {acc_total:.4f} ({acc_total*100:.2f}%)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cm_aggregate.png'), dpi=150)
    plt.show()
    plt.close()

    print(f"\nOutput tersimpan di: {output_dir}/")
    print(f"  cm_fold1.png ... cm_fold5.png → confusion matrix per fold")
    print(f"  cm_all_folds.png              → semua fold dalam 1 gambar")
    print(f"  cm_aggregate.png              → gabungan semua fold")
    print(f"Overall Accuracy (agregat): {acc_total:.4f} ({acc_total*100:.2f}%)")

    return all_fold_cms, cm_total

# ============================================================
# JALANKAN KEDUA SKENARIO
# ============================================================
cms_imb, cm_imb_total = run_cv_confusion_matrix(
    data_path     = 'dataset/Imbalanced',
    skenario_name = 'Imbalanced',
    output_dir    = 'results/cv_imbalanced/confusion_matrix'
)

cms_bal, cm_bal_total = run_cv_confusion_matrix(
    data_path     = 'dataset/Balanced_GAN',
    skenario_name = 'Balanced GAN',
    output_dir    = 'results/cv_balanced_gan/confusion_matrix'
)

print("\n====== SEMUA SELESAI ======")
print("Output ada di:")
print("  results/cv_imbalanced/confusion_matrix/")
print("  results/cv_balanced_gan/confusion_matrix/")