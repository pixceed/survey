import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn # ネットワークの構築
import torch.nn.functional as F # 様々な関数の使用
import torch.optim as optim # 最適化アルゴリズムの使用
import torchvision # 画像処理に関係する処理の使用
import torchvision.transforms as transforms # 画像変換機能の使用


class Net(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            # 畳み込み層
            # 1チャンネルを32チャンネルにする、3x3のフィルターを使う、1つずつずらす
            nn.Conv2d(1, 32, 3, 1),
            # 活性化関数
            nn.ReLU(),
            # プーリング層、2x2の領域から最大のものを1つ取り出す
            nn.MaxPool2d(2, 2),
            # Dropout
            nn.Dropout(0.1),
        )
        self.layer2 = nn.Sequential(
            # 畳み込み層
            # 32チャンネルを64チャンネルにする、3x3のフィルターを使う、1つずつずらす
            nn.Conv2d(32, 64, 3, 1),
            # 活性化関数
            nn.ReLU(),
            # プーリング層、2x2の領域から最大のものを1つ取り出す
            nn.MaxPool2d(2, 2),
            # Dropout
            nn.Dropout(0.1),
        )
        self.layer3 = nn.Sequential(
            # (チャンネル数 x 縦 x 横)を1次元に変換する
            nn.Flatten(),
            # 線形層
            nn.Linear(64*5*5, 256),
            # 活性化関数
            nn.ReLU(),
            # 線形層
            nn.Linear(256, 10),
            # 出力層
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

model_path = "src/03_mnist_conv/output/mnist_99.pth"
model = torch.load(model_path)
model.eval()

# 検証データ
test_dataset = torchvision.datasets.MNIST(root='src/03_mnist_conv/input',
                                        train=False,
                                        transform=transforms.ToTensor(),
                                        download = True)

batch_size = 256

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                        batch_size=batch_size,
                                        shuffle=True)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

plt.figure(figsize=(20, 10))
# for i in range(10):
with torch.no_grad():
    all_acc = 0
    for i, (x, t) in enumerate(test_loader):

        if i == 10:
            break

        x = x.to(device)
        t = t.to(device)

        y = model(x)
        y_label = torch.argmax(y, dim=1)

        acc = torch.sum(y_label == t) * 1.0 / len(t)
        all_acc += acc

        ax = plt.subplot(1, 10, i+1)

        rn = random.randint(0, 255)

        plt.imshow(x[rn].detach().to('cpu').numpy().reshape(28, 28), cmap='gray')
        ax.axis('off')
        ax.set_title('label : {}\n Prediction : {}'.format(t[rn], y_label[rn]), fontsize=15)
    plt.show()

