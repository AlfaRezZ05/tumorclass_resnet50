import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)
from sklearn.preprocessing import label_binarize
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE   = 224
BATCH_SIZE = 32
TEST_PATH  = 'dataset/Testing'

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def evaluate_model(model_path, train_path, skenario_name):
    print(f"\nEvaluasi: {skenario_name}")

    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    test_dataset  = datasets.ImageFolder(TEST_PATH,  transform=transform)
    class_names   = train_dataset.classes
    num_classes   = len(class_names)
    test_loader   = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs    = imgs.to(DEVICE)
            outputs = model(imgs)
            probs   = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    accuracy  = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall    = recall_score(all_labels, all_preds, average='macro')
    f1        = f1_score(all_labels, all_preds, average='macro')

    specificities = []
    for i in range(num_classes):
        tn = np.sum((all_labels != i) & (all_preds != i))
        fp = np.sum((all_labels != i) & (all_preds == i))
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    specificity = np.mean(specificities)

    y_bin = label_binarize(all_labels, classes=range(num_classes))
    auc   = roc_auc_score(y_bin, all_probs, average='macro', multi_class='ovr')

    precision_per = precision_score(all_labels, all_preds, average=None)
    recall_per    = recall_score(all_labels, all_preds, average=None)
    f1_per        = f1_score(all_labels, all_preds, average=None)
    acc_per       = [accuracy_score(all_labels == i, all_preds == i)
                     for i in range(num_classes)]
    spec_per      = specificities

    print(f"  Accuracy   : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  F1-Score   : {f1:.4f}")
    print(f"  AUC        : {auc:.4f}")

    return {
        'skenario'      : skenario_name,
        'accuracy'      : accuracy,
        'precision'     : precision,
        'recall'        : recall,
        'specificity'   : specificity,
        'f1_score'      : f1,
        'auc'           : auc,
        'class_names'   : class_names,
        'precision_per' : precision_per,
        'recall_per'    : recall_per,
        'f1_per'        : f1_per,
        'acc_per'       : acc_per,
        'spec_per'      : spec_per,
    }

# Skenario 1: Imbalanced
hasil_imb = evaluate_model(
    model_path    = 'results/imbalanced/resnet50_model.pth',
    train_path    = 'dataset/Imbalanced',
    skenario_name = 'Imbalanced'
)

# Skenario 2: Balanced GAN
hasil_bal = evaluate_model(
    model_path    = 'results/balanced_gan/resnet50_model.pth',
    train_path    = 'dataset/Balanced_GAN',
    skenario_name = 'Balanced GAN'
)

# Tabel Overall
print("\n" + "="*65)
print("TABEL PERBANDINGAN IMBALANCED VS BALANCED GAN")
print("="*65)
metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'auc']
labels  = ['Accuracy', 'Precision', 'Recall (Sensitivity)', 'Specificity', 'F1-Score', 'AUC']
print(f"\n{'Metrik':<25} {'Imbalanced':>12} {'Balanced GAN':>14} {'Selisih':>10}")
print("-" * 65)
for m, l in zip(metrics, labels):
    imb = hasil_imb[m]
    bal = hasil_bal[m]
    sel = bal - imb
    tanda = '+' if sel >= 0 else ''
    print(f"{l:<25} {imb:>12.4f} {bal:>14.4f} {tanda}{sel:>9.4f}")

# Tabel Per Kelas
print("\n" + "="*70)
print("PERBANDINGAN PER KELAS")
print("="*70)
class_names = hasil_imb['class_names']
for i, kelas in enumerate(class_names):
    print(f"\n--- {kelas} ---")
    print(f"{'Metrik':<15} {'Imbalanced':>12} {'Balanced GAN':>14}")
    print("-" * 42)
    print(f"{'Accuracy':<15} {hasil_imb['acc_per'][i]:>12.4f} {hasil_bal['acc_per'][i]:>14.4f}")
    print(f"{'Precision':<15} {hasil_imb['precision_per'][i]:>12.4f} {hasil_bal['precision_per'][i]:>14.4f}")
    print(f"{'Recall':<15} {hasil_imb['recall_per'][i]:>12.4f} {hasil_bal['recall_per'][i]:>14.4f}")
    print(f"{'Specificity':<15} {hasil_imb['spec_per'][i]:>12.4f} {hasil_bal['spec_per'][i]:>14.4f}")
    print(f"{'F1-Score':<15} {hasil_imb['f1_per'][i]:>12.4f} {hasil_bal['f1_per'][i]:>14.4f}")

# Simpan Excel
df_overall = pd.DataFrame({
    'Metrik'       : labels,
    'Imbalanced'   : [hasil_imb[m] for m in metrics],
    'Balanced GAN' : [hasil_bal[m] for m in metrics],
    'Selisih'      : [hasil_bal[m] - hasil_imb[m] for m in metrics]
})
df_imb = pd.DataFrame({
    'Kelas'      : class_names,
    'Accuracy'   : hasil_imb['acc_per'],
    'Precision'  : hasil_imb['precision_per'],
    'Recall'     : hasil_imb['recall_per'],
    'Specificity': hasil_imb['spec_per'],
    'F1-Score'   : hasil_imb['f1_per'],
})
df_bal = pd.DataFrame({
    'Kelas'      : class_names,
    'Accuracy'   : hasil_bal['acc_per'],
    'Precision'  : hasil_bal['precision_per'],
    'Recall'     : hasil_bal['recall_per'],
    'Specificity': hasil_bal['spec_per'],
    'F1-Score'   : hasil_bal['f1_per'],
})

with pd.ExcelWriter('hasil_evaluasi_gan.xlsx', engine='openpyxl') as writer:
    df_overall.to_excel(writer, index=False, sheet_name='Perbandingan Overall')
    df_imb.to_excel(writer, index=False, sheet_name='Imbalanced per Kelas')
    df_bal.to_excel(writer, index=False, sheet_name='Balanced GAN per Kelas')
    for sheet in writer.sheets.values():
        for col in sheet.columns:
            max_len = max(len(str(cell.value)) if cell.value else 0 for cell in col)
            sheet.column_dimensions[col[0].column_letter].width = max_len + 4

print("\nFile Excel tersimpan: hasil_evaluasi_gan.xlsx")
print("\n====== SELESAI ======")