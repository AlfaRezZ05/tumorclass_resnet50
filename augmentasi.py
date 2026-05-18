import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
import shutil

# ============================================================
# PATH
# ============================================================
train_path        = 'Training'
output_imbalanced = 'Imbalanced'
output_balanced   = 'Balanced'

TARGET_SIZE = (224, 224)  # ukuran resize

# ============================================================
# FUNGSI AUGMENTASI
# ============================================================
def augment_image(img):
    """Lakukan augmentasi random pada 1 gambar"""
    augmentations = []

    # 1. Rotate
    angle = random.choice([90, 180, 270, 45, -45])
    augmentations.append(img.rotate(angle, expand=True))

    # 2. Flip horizontal
    augmentations.append(img.transpose(Image.FLIP_LEFT_RIGHT))

    # 3. Flip vertical
    augmentations.append(img.transpose(Image.FLIP_TOP_BOTTOM))

    # 4. Brightness
    enhancer = ImageEnhance.Brightness(img)
    augmentations.append(enhancer.enhance(random.uniform(0.7, 1.3)))

    # 5. Contrast
    enhancer = ImageEnhance.Contrast(img)
    augmentations.append(enhancer.enhance(random.uniform(0.7, 1.3)))

    # 6. Crop & resize
    width, height = img.size
    left   = random.randint(0, width  // 6)
    top    = random.randint(0, height // 6)
    right  = width  - random.randint(0, width  // 6)
    bottom = height - random.randint(0, height // 6)
    augmentations.append(img.crop((left, top, right, bottom)))

    # Pilih 1 augmentasi random
    result = random.choice(augmentations)
    return result.resize(TARGET_SIZE)

def resize_image(img):
    """Hanya resize tanpa augmentasi"""
    return img.resize(TARGET_SIZE)

# ============================================================
# FUNGSI COPY & RESIZE FOLDER
# ============================================================
def copy_and_resize(src_folder, dst_folder):
    """Copy semua gambar dari src ke dst sambil di-resize"""
    os.makedirs(dst_folder, exist_ok=True)
    files = [f for f in os.listdir(src_folder)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for fname in files:
        img = Image.open(os.path.join(src_folder, fname)).convert('RGB')
        img = resize_image(img)
        img.save(os.path.join(dst_folder, fname))
    return len(files)

# ============================================================
# FUNGSI GENERATE SAMPAI TARGET
# ============================================================
def generate_until_target(src_folder, dst_folder, target):
    """Copy asli + generate augmentasi sampai jumlah = target"""
    os.makedirs(dst_folder, exist_ok=True)

    # Copy gambar asli dulu
    files = [f for f in os.listdir(src_folder)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for fname in files:
        img = Image.open(os.path.join(src_folder, fname)).convert('RGB')
        img = resize_image(img)
        img.save(os.path.join(dst_folder, fname))

    current = len(files)
    count   = 0

    # Generate augmentasi sampai target
    while current < target:
        fname = random.choice(files)
        img   = Image.open(os.path.join(src_folder, fname)).convert('RGB')
        aug   = augment_image(img)
        aug.save(os.path.join(dst_folder, f'aug_{count:05d}.jpg'))
        current += 1
        count   += 1

    return current

# ============================================================
# MAIN
# ============================================================
kelas_list = sorted([k for k in os.listdir(train_path)
                     if os.path.isdir(os.path.join(train_path, k))])

# Hitung jumlah asli
jumlah_asli = {k: len([f for f in os.listdir(os.path.join(train_path, k))
                        if f.lower().endswith(('.jpg','.jpeg','.png'))])
               for k in kelas_list}

TARGET_BALANCED = max(jumlah_asli.values())  # samakan ke kelas terbanyak
print(f"Target balanced: {TARGET_BALANCED} per kelas\n")

# ============================================================
# SKENARIO 1: IMBALANCED (hanya resize)
# ============================================================
print("=" * 50)
print("SKENARIO 1: IMBALANCED (resize only)")
print("=" * 50)
for kelas in kelas_list:
    src = os.path.join(train_path, kelas)
    dst = os.path.join(output_imbalanced, kelas)
    n   = copy_and_resize(src, dst)
    print(f"  {kelas:<25} {n} gambar")

# ============================================================
# SKENARIO 2: BALANCED (resize + augmentasi)
# ============================================================
print("\n" + "=" * 50)
print("SKENARIO 2: BALANCED (resize + augmentasi)")
print("=" * 50)
for kelas in kelas_list:
    src    = os.path.join(train_path, kelas)
    dst    = os.path.join(output_balanced, kelas)
    jumlah = jumlah_asli[kelas]

    if jumlah < TARGET_BALANCED:
        print(f"  {kelas:<25} {jumlah} -> generate sampai {TARGET_BALANCED}...")
        n = generate_until_target(src, dst, TARGET_BALANCED)
    else:
        n = copy_and_resize(src, dst)

    print(f"  {kelas:<25} selesai: {n} gambar")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 50)
print("RINGKASAN HASIL")
print("=" * 50)
print(f"{'Kelas':<25} {'Imbalanced':>12} {'Balanced':>12}")
print("-" * 50)
for kelas in kelas_list:
    n_imb = len(os.listdir(os.path.join(output_imbalanced, kelas)))
    n_bal = len(os.listdir(os.path.join(output_balanced,   kelas)))
    print(f"{kelas:<25} {n_imb:>12} {n_bal:>12}")

print("\n====== SELESAI ======")
print("Folder output:")
print(f"  - {output_imbalanced}")
print(f"  - {output_balanced}")