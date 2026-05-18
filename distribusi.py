import os
import matplotlib.pyplot as plt
import seaborn as sns

# Path dataset
train_path = 'Training'
test_path = 'Testing'

# Hitung jumlah gambar per kelas
def hitung_distribusi(path):
    distribusi = {}
    for kelas in os.listdir(path):
        kelas_path = os.path.join(path, kelas)
        if os.path.isdir(kelas_path):
            distribusi[kelas] = len(os.listdir(kelas_path))
    return distribusi

train_dist = hitung_distribusi(train_path)
test_dist = hitung_distribusi(test_path)

# Print tabel distribusi
print("\n=== DISTRIBUSI DATA ===")
print(f"{'Kelas':<20} {'Train':>10} {'Test':>10} {'Total':>10}")
print("-" * 50)
for kelas in train_dist:
    train = train_dist[kelas]
    test = test_dist.get(kelas, 0)
    total = train + test
    print(f"{kelas:<20} {train:>10} {test:>10} {total:>10}")

# Visualisasi grafik
kelas = list(train_dist.keys())
train_vals = list(train_dist.values())
test_vals = [test_dist.get(k, 0) for k in kelas]

x = range(len(kelas))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar([i - width/2 for i in x], train_vals, width, label='Train', color='steelblue')
bars2 = ax.bar([i + width/2 for i in x], test_vals, width, label='Test', color='orange')

ax.set_xlabel('Kelas')
ax.set_ylabel('Jumlah Gambar')
ax.set_title('Distribusi Data Brain Tumor MRI')
ax.set_xticks(x)
ax.set_xticklabels(kelas, rotation=15)
ax.legend()

# Tambah angka di atas bar
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(int(bar.get_height())), ha='center', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(int(bar.get_height())), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('distribusi_data.png', dpi=150)
plt.show()
print("\nGrafik tersimpan di distribusi_data.png")