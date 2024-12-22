import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

# ================================================
# 1. パラメータとデバイス
# ================================================
IMAGE_SIZE = 32     # もとは28x28 → パディングで32x32に
CHANNELS = 1        # グレースケール
BATCH_SIZE = 100
EMBEDDING_DIM = 2   # 潜在空間の次元
EPOCHS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ================================================
# 2. データの準備 (FashionMNIST)
# ================================================
# 28x28 -> パディングで32x32, テンソル化(0~1)
transform = transforms.Compose([
    transforms.Pad(2),  # 28→32
    transforms.ToTensor()
])

# ダウンロード & ロード
train_dataset = torchvision.datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# NumPy形式 (主に可視化用/潜在空間プロット用などに使用)
x_train_all = train_dataset.data.numpy()  # (60000, 28, 28) - ただしパディング前
y_train_all = train_dataset.targets.numpy()  # (60000,)
x_test_all = test_dataset.data.numpy()   # (10000, 28, 28)
y_test_all = test_dataset.targets.numpy()  # (10000,)


# ================================================
# 3. モデル定義 (Encoder / Decoder / Autoencoder)
# ================================================
class Encoder(nn.Module):
    def __init__(self, embedding_dim=2):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)   # 32x32 -> 16x16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 16x16 -> 8x8
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # 8x8 -> 4x4
        self.fc = nn.Linear(128 * 4 * 4, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))  # (N, 1, 32, 32) -> (N, 32, 16, 16)
        x = self.relu(self.conv2(x))  # -> (N, 64, 8, 8)
        x = self.relu(self.conv3(x))  # -> (N, 128, 4, 4)
        x = x.view(x.size(0), -1)     # (N, 128*4*4)
        x = self.fc(x)                # (N, embedding_dim)
        return x


class Decoder(nn.Module):
    def __init__(self, embedding_dim=2):
        super(Decoder, self).__init__()
        # 潜在ベクトル -> 128x4x4
        self.fc = nn.Linear(embedding_dim, 128 * 4 * 4)

        # ConvTransposeでアップサンプリング
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,
                                          padding=1, output_padding=1)  # 4->8
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                                          padding=1, output_padding=1)  # 8->16
        self.deconv3 = nn.ConvTranspose2d(32, 128, kernel_size=3, stride=2,
                                          padding=1, output_padding=1)  # 16->32
        # 最後に (128->1, stride=1) で32→32を維持
        self.deconv_final = nn.ConvTranspose2d(128, 1, kernel_size=3,
                                               stride=1, padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)                # (N, 128*4*4)
        x = x.view(x.size(0), 128, 4, 4)

        x = self.relu(self.deconv1(x))   # 4->8
        x = self.relu(self.deconv2(x))   # 8->16
        x = self.relu(self.deconv3(x))   # 16->32
        x = self.deconv_final(x)         # (128->1), stride=1 => 32->32
        x = self.sigmoid(x)             # 出力: (N, 1, 32, 32)
        return x


class Autoencoder(nn.Module):
    def __init__(self, embedding_dim=2):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(embedding_dim)
        self.decoder = Decoder(embedding_dim)

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed


# ================================================
# 4. 学習 (Train)
# ================================================
autoencoder = Autoencoder(EMBEDDING_DIM).to(device)
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
criterion = nn.BCELoss()  # 最終層にSigmoidを使用しているためBCELoss

print("\n===== Model Summary =====")
print(autoencoder, "\n")

for epoch in range(EPOCHS):
    autoencoder.train()
    running_loss = 0.0

    for images, _ in train_loader:
        images = images.to(device)  # (N,1,32,32)
        optimizer.zero_grad()

        reconstructed = autoencoder(images)
        loss = criterion(reconstructed, images)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")


# ================================================
# 5. 再構成の可視化
# ================================================
autoencoder.eval()

def show_images(imgs, title="", cmap="gray"):
    """imgs: shape=(N, 1, H, W) のnumpy配列を1列で表示"""
    fig = plt.figure(figsize=(14, 2))
    fig.suptitle(title)
    for i in range(len(imgs)):
        ax = fig.add_subplot(1, len(imgs), i + 1)
        ax.axis("off")
        ax.imshow(imgs[i, 0], cmap=cmap)

# テストデータからバッチを1つ取り出して最初のn_to_predict枚を可視化
n_to_predict = 10
test_iter = iter(test_loader)
test_images, test_labels = next(test_iter)  # バッチ(100枚)
test_images = test_images[:n_to_predict].to(device)

with torch.no_grad():
    test_recon = autoencoder(test_images)

origin = test_images.cpu().numpy()  # (n_to_predict,1,32,32)
recon = test_recon.cpu().numpy()

print("\nExample real clothing items")
show_images(origin, title="Original")
print("Reconstructions")
show_images(recon, title="Reconstructed")
plt.show()


# ================================================
# 6. 潜在空間(Embedding)の可視化
# ================================================
# テストセット(例: 5000枚)を使って埋め込み(z)を取得
n_to_embed = 5000
x_test_np = x_test_all[:n_to_embed]   # (n_to_embed,28,28)
y_test_np = y_test_all[:n_to_embed]

# 32x32にPadしてTensor化
x_test_tensor = torch.from_numpy(x_test_np).unsqueeze(1).float() / 255.0
x_test_tensor = torch.nn.functional.pad(x_test_tensor, (2,2,2,2))  # 上下左右に2ピクセルずつ
x_test_tensor = x_test_tensor.to(device)

with torch.no_grad():
    z = autoencoder.encoder(x_test_tensor)  # (n_to_embed, EMBEDDING_DIM)
z = z.cpu().numpy()

# 潜在空間を散布図に
plt.figure(figsize=(8,8))
plt.scatter(z[:, 0], z[:, 1], c="black", alpha=0.5, s=3)
plt.title("Latent Space (all in black)")
plt.show()

# ラベルで色分け
plt.figure(figsize=(8,8))
plt.scatter(z[:, 0], z[:, 1], c=y_test_np, cmap="rainbow", alpha=0.8, s=3)
plt.colorbar()
plt.title("Latent Space (colored by label)")
plt.show()


# ================================================
# 7. 潜在空間からのサンプリング (Decoderで生成)
# ================================================
# 潜在空間 z の最小値・最大値
mins = np.min(z, axis=0)
maxs = np.max(z, axis=0)

# ランダムサンプリング
grid_width, grid_height = 6, 3
samples = np.random.uniform(mins, maxs, size=(grid_width * grid_height, EMBEDDING_DIM))

# テンソル化してDecoderへ
samples_tensor = torch.from_numpy(samples).float().to(device)
autoencoder.eval()
with torch.no_grad():
    gen_recon = autoencoder.decoder(samples_tensor)

gen_recon = gen_recon.cpu().numpy()  # (grid_width*grid_height,1,32,32)

# 元の潜在空間とサンプル点の比較
plt.figure(figsize=(8,6))
plt.scatter(z[:, 0], z[:, 1], c="black", alpha=0.5, s=2, label="Original embeddings")
plt.scatter(samples[:, 0], samples[:, 1], c="#00B0F0", alpha=1, s=40, label="Random samples")
plt.title("Latent Space + Sampled Points")
plt.legend()
plt.show()

# 生成された画像をタイル状に表示
fig = plt.figure(figsize=(8, grid_height*2))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(grid_width * grid_height):
    ax = fig.add_subplot(grid_height, grid_width, i + 1)
    ax.axis("off")
    ax.text(
        0.5, -0.35, str(np.round(samples[i, :], 1)),
        fontsize=10, ha="center", transform=ax.transAxes,
    )
    ax.imshow(gen_recon[i, 0], cmap="Greys")
plt.show()


# ================================================
# 8. 潜在空間をグリッド状にサンプリング (追加の可視化例)
# ================================================
grid_size = 15
x_vals = np.linspace(mins[0], maxs[0], grid_size)
y_vals = np.linspace(mins[1], maxs[1], grid_size)
xv, yv = np.meshgrid(x_vals, y_vals)
grid_points = np.stack([xv.flatten(), yv.flatten()], axis=1)  # (grid_size^2,2)

grid_tensor = torch.from_numpy(grid_points).float().to(device)
with torch.no_grad():
    grid_recon = autoencoder.decoder(grid_tensor)
grid_recon = grid_recon.cpu().numpy()  # (grid_size^2,1,32,32)

# タイル表示
fig = plt.figure(figsize=(12, 12))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(grid_size**2):
    ax = fig.add_subplot(grid_size, grid_size, i + 1)
    ax.axis("off")
    ax.imshow(grid_recon[i, 0], cmap="Greys")
plt.show()
