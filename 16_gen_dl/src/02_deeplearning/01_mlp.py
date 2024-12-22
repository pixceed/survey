

import torch
import torch.nn as nn
import torch.optim as optim
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
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 3, 200)  # 画像は 32x32x3 チャネル
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(200, 150)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(150, NUM_CLASSES)
        # 学習時の loss 関数が CrossEntropyLoss の場合、最後に softmax は不要

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x  # softmax は学習時には不要 (CrossEntropyLoss が内部で処理)

model = MLP()

# ======================
# 3. Train the model
# ======================
# 損失関数とオプティマイザを定義
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        # 勾配の初期化



        
        optimizer.zero_grad()

        # 順伝搬
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 逆伝搬
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

# ======================
# 4. Evaluation
# ======================
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        # outputs は (バッチサイズ, 10) の形状なので、argmax(dim=1) でクラスを推定
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100.0 * correct / total
print(f"Accuracy of the model on the test images: {accuracy:.2f}%")

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

# 予測を可視化する数
n_to_show = 10
indices = np.random.choice(range(len(test_dataset)), n_to_show, replace=False)

fig = plt.figure(figsize=(15, 3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, idx in enumerate(indices):
    img, label = test_dataset[idx]
    # (C, H, W) -> モデルに入力する際は (1, C, H, W) に拡張
    img_input = img.unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_input)
        pred = outputs.argmax(dim=1).item()

    ax = fig.add_subplot(1, n_to_show, i + 1)
    ax.axis("off")
    ax.imshow(np.transpose(img.numpy(), (1, 2, 0)))  # (C, H, W) -> (H, W, C)
    ax.text(0.5, -0.35, "pred = " + CLASSES[pred], fontsize=10, ha="center", transform=ax.transAxes)
    ax.text(0.5, -0.7, "act = " + CLASSES[label], fontsize=10, ha="center", transform=ax.transAxes)

plt.show()
