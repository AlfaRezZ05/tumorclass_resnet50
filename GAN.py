import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURASI
# ============================================================
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE    = 64        # GAN pakai 64x64, nanti di-resize ke 224x224
LATENT_DIM  = 100       # Dimensi noise input Generator
BATCH_SIZE  = 32
EPOCHS_GAN  = 200       # Epoch training GAN
LR          = 0.0002
BETA1       = 0.5
TARGET_SIZE = 827       # Target jumlah per kelas (ikut kelas terbanyak)

print(f"Device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")

# ============================================================
# DATASET LOADER PER KELAS
# ============================================================
class SingleClassDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files     = [os.path.join(folder_path, f)
                          for f in os.listdir(folder_path)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

transform_gan = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ============================================================
# ARSITEKTUR DCGAN
# ============================================================

# --- Generator ---
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# --- Discriminator ---
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1).squeeze(1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# ============================================================
# FUNGSI TRAINING GAN
# ============================================================
def train_gan(class_folder, class_name, epochs=EPOCHS_GAN):
    print(f"\n{'='*55}")
    print(f"Training DCGAN: {class_name}")
    print(f"{'='*55}")

    os.makedirs(f'gan_checkpoints/{class_name}', exist_ok=True)
    os.makedirs(f'gan_samples/{class_name}', exist_ok=True)

    dataset    = SingleClassDataset(class_folder, transform=transform_gan)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=0)
    print(f"Jumlah gambar asli: {len(dataset)}")

    G = Generator(LATENT_DIM).to(DEVICE)
    D = Discriminator().to(DEVICE)
    G.apply(weights_init)
    D.apply(weights_init)

    criterion   = nn.BCELoss()
    opt_G       = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))
    opt_D       = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))
    fixed_noise = torch.randn(16, LATENT_DIM, 1, 1, device=DEVICE)

    G_losses, D_losses = [], []

    for epoch in range(epochs):
        g_loss_epoch, d_loss_epoch = 0, 0

        for real_imgs in dataloader:
            real_imgs   = real_imgs.to(DEVICE)
            b_size      = real_imgs.size(0)
            real_labels = torch.ones(b_size,  device=DEVICE)
            fake_labels = torch.zeros(b_size, device=DEVICE)

            # Train Discriminator
            D.zero_grad()
            loss_real = criterion(D(real_imgs), real_labels)
            noise     = torch.randn(b_size, LATENT_DIM, 1, 1, device=DEVICE)
            fake_imgs = G(noise)
            loss_fake = criterion(D(fake_imgs.detach()), fake_labels)
            loss_D    = loss_real + loss_fake
            loss_D.backward()
            opt_D.step()

            # Train Generator
            G.zero_grad()
            loss_G = criterion(D(fake_imgs), real_labels)
            loss_G.backward()
            opt_G.step()

            g_loss_epoch += loss_G.item()
            d_loss_epoch += loss_D.item()

        avg_g = g_loss_epoch / len(dataloader)
        avg_d = d_loss_epoch / len(dataloader)
        G_losses.append(avg_g)
        D_losses.append(avg_d)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:03d}/{epochs}] "
                  f"Loss_D: {avg_d:.4f} | Loss_G: {avg_g:.4f}")
            with torch.no_grad():
                sample = G(fixed_noise)
                save_image(sample * 0.5 + 0.5,
                           f'gan_samples/{class_name}/epoch_{epoch+1:03d}.png',
                           nrow=4)

    # Simpan model & grafik loss
    torch.save(G.state_dict(),
               f'gan_checkpoints/{class_name}/generator.pth')

    plt.figure(figsize=(10, 4))
    plt.plot(G_losses, label='Generator Loss',     color='blue')
    plt.plot(D_losses, label='Discriminator Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'DCGAN Training Loss - {class_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'gan_checkpoints/{class_name}/gan_loss.png', dpi=150)
    plt.close()
    print(f"Model & grafik loss tersimpan di gan_checkpoints/{class_name}/")

    return G

# ============================================================
# FUNGSI GENERATE GAMBAR SINTETIS
# ============================================================
def generate_synthetic(G, class_folder, class_name,
                        output_dir, target=TARGET_SIZE):
    print(f"\nGenerating gambar sintetis: {class_name}")
    os.makedirs(output_dir, exist_ok=True)

    # Copy gambar asli
    real_files = [f for f in os.listdir(class_folder)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for fname in real_files:
        img = Image.open(os.path.join(class_folder, fname)).convert('RGB')
        img = img.resize((224, 224))
        img.save(os.path.join(output_dir, fname))

    current = len(real_files)
    needed  = target - current
    print(f"Gambar asli: {current} | Perlu generate: {needed}")

    if needed <= 0:
        print("Sudah mencukupi target!")
        return current

    # Generate sintetis dari GAN
    G.eval()
    count = 0
    with torch.no_grad():
        while count < needed:
            batch = min(BATCH_SIZE, needed - count)
            noise = torch.randn(batch, LATENT_DIM, 1, 1, device=DEVICE)
            fake  = G(noise) * 0.5 + 0.5

            for i in range(batch):
                img_pil = transforms.ToPILImage()(fake[i].cpu())
                img_pil = img_pil.resize((224, 224))
                img_pil.save(os.path.join(output_dir,
                                          f'gan_syn_{count:05d}.jpg'))
                count += 1

    total = len(os.listdir(output_dir))
    print(f"Total: {total} (asli: {current} + sintetis GAN: {count})")
    return total

# ============================================================
# MAIN
# ============================================================
train_path          = 'dataset/Training'
output_balanced_gan = 'dataset/Balanced_GAN'

kelas_list  = sorted(os.listdir(train_path))
jumlah_asli = {k: len([f for f in os.listdir(os.path.join(train_path, k))
                        if f.lower().endswith(('.jpg','.jpeg','.png'))])
               for k in kelas_list}

print("=" * 55)
print("DISTRIBUSI DATA & KEBUTUHAN GAN")
print("=" * 55)
print(f"{'Kelas':<25} {'Jumlah':>8} {'Target':>8} {'Perlu':>8}")
print("-" * 55)
for k, v in jumlah_asli.items():
    needed = max(0, TARGET_SIZE - v)
    status = '← GAN' if needed > 0 else '✓ OK'
    print(f"{k:<25} {v:>8} {TARGET_SIZE:>8} {needed:>8}  {status}")

print()

# Training GAN & Generate per kelas
for kelas in kelas_list:
    src_folder = os.path.join(train_path, kelas)
    dst_folder = os.path.join(output_balanced_gan, kelas)
    jumlah     = jumlah_asli[kelas]

    if jumlah < TARGET_SIZE:
        # Train GAN & generate sintetis
        G = train_gan(src_folder, kelas, epochs=EPOCHS_GAN)
        generate_synthetic(G, src_folder, kelas, dst_folder, TARGET_SIZE)
    else:
        # Copy & resize saja
        print(f"\n>>> {kelas}: {jumlah} gambar → copy & resize")
        os.makedirs(dst_folder, exist_ok=True)
        files = [f for f in os.listdir(src_folder)
                 if f.lower().endswith(('.jpg','.jpeg','.png'))]
        for fname in files:
            img = Image.open(os.path.join(src_folder, fname)).convert('RGB')
            img = img.resize((224, 224))
            img.save(os.path.join(dst_folder, fname))
        print(f"    Selesai: {len(files)} gambar")

# Ringkasan
print("\n" + "=" * 55)
print("RINGKASAN HASIL BALANCED GAN")
print("=" * 55)
print(f"{'Kelas':<25} {'Asli':>8} {'Hasil':>8}")
print("-" * 55)
for kelas in kelas_list:
    dst = os.path.join(output_balanced_gan, kelas)
    n   = len(os.listdir(dst)) if os.path.exists(dst) else 0
    print(f"{kelas:<25} {jumlah_asli[kelas]:>8} {n:>8}")

print("\n====== SELESAI ======")
print("Output tersimpan di:")
print(f"  - {output_balanced_gan}/     ← data untuk training")
print(f"  - gan_samples/               ← contoh gambar GAN tiap epoch")
print(f"  - gan_checkpoints/           ← model GAN & grafik loss")