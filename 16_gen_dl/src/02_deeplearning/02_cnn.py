import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

# ======================
# 0. Parameters
# ======================
NUM_CLASSES = 10
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0005

# ======================
# 1. Prepare the Data
# ======================
# CIFAR-10 データをダウンロード & ロード
# transforms.ToTensor() で画像を [0, 1] 区間に正規化＆PyTorchのテンソルに変換
transform = transforms.ToTensor()

train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ======================
# 2. Build the model
# ======================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # [batch, 3, 32, 32]

        # block1: Conv2D -> BatchNorm -> LeakyReLU
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, 
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.lrelu1 = nn.LeakyReLU()

        # block2: Conv2D -> BatchNorm -> LeakyReLU (stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, 
                               kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.lrelu2 = nn.LeakyReLU()

        # block3: Conv2D -> BatchNorm -> LeakyReLU
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.lrelu3 = nn.LeakyReLU()

        # block4: Conv2D -> BatchNorm -> LeakyReLU (stride=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.lrelu4 = nn.LeakyReLU()

        # 全結合層の前に Flatten を行う (forward で実装してもOK)
        # self.flatten = nn.Flatten()

        # 全結合層 (128次元) -> BatchNorm -> LeakyReLU -> Dropout
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # 32x32 -> conv(stride=2) -> 16x16 -> conv(stride=2) -> 8x8
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.lrelu_fc1 = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.5)

        # 出力層
        self.fc2 = nn.Linear(128, NUM_CLASSES)
        # 学習時の loss 関数が CrossEntropyLoss の場合、forward 内では softmax 不要

    def forward(self, x):
        # (batch, 3, 32, 32) -> block1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lrelu1(x)

        # block2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu2(x)

        # block3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lrelu3(x)

        # block4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.lrelu4(x)

        # Flatten
        x = x.view(x.size(0), -1)  # (batch, 64*8*8)

        # 全結合層 (BN -> LeakyReLU -> Dropout)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.lrelu_fc1(x)
        x = self.dropout(x)

        # 出力層 (logits)
        x = self.fc2(x)
        return x

model = CNN()

# ======================
# 3. Train the model
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# GPU が使える場合は .cuda() を付与 (任意)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Epochごとの平均 loss を表示
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

# ======================
# 4. Evaluation
# ======================
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        # 出力 (batch, 10) から予測ラベルを取得
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100.0 * correct / total
print(f"Accuracy of the model on the test images: {accuracy:.2f}%")

# (追加) バリデーションのように test_loader で valid_loss も見る例
# ※ TensorFlow版の「validation_data=(x_test, y_test)」に相当
# ここでは簡単に精度のみ計算し、valid_loss は省略

# ======================
# 5. Check some predictions
# ======================
CLASSES = np.array(
    [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
)

# PyTorch Tensor -> NumPy で可視化
# x_test, y_test は手元に NumPy 形式でないため、dataset/test_loader から取得
test_data = test_dataset.data  # shape=(10000, 32, 32, 3), uint8
test_labels = np.array(test_dataset.targets)

# モデルの出力を得るには、test_loader から一括で推論するか、個別に行う方法があります。
# ここでは個別に行います (大きいデータには非推奨)
# ただし、メモリに余裕があれば全データをバッチ推論してOKです。

n_to_show = 10
indices = np.random.choice(range(len(test_dataset)), n_to_show, replace=False)

fig = plt.figure(figsize=(15, 3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, idx in enumerate(indices):
    # 画像データを [0,1] に正規化して Tensor へ
    img_arr = test_data[idx].astype("float32") / 255.0  # (32, 32, 3)
    label = test_labels[idx]
    # (H, W, C) -> (C, H, W)
    img_tensor = torch.from_numpy(np.transpose(img_arr, (2, 0, 1)))  
    img_tensor = img_tensor.unsqueeze(0).to(device)  # バッチ次元追加

    with torch.no_grad():
        outputs = model(img_tensor)
        pred = outputs.argmax(dim=1).item()

    ax = fig.add_subplot(1, n_to_show, i + 1)
    ax.axis("off")

    # 表示用に (32,32,3) へ戻す
    ax.imshow(img_arr)
    ax.text(0.5, -0.35, f"pred = {CLASSES[pred]}", fontsize=10, ha="center", transform=ax.transAxes)
    ax.text(0.5, -0.7,  f"act  = {CLASSES[label]}", fontsize=10, ha="center", transform=ax.transAxes)

plt.show()
