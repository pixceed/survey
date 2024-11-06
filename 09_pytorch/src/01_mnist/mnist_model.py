import os
import numpy as np
import matplotlib.pyplot as plt

from pprint import pprint

import torch
import torch.nn as nn # ネットワークの構築
import torch.nn.functional as F # 様々な関数の使用
import torch.optim as optim # 最適化アルゴリズムの使用
import torchvision # 画像処理に関係する処理の使用
import torchvision.transforms as transforms # 画像変換機能の使用



# ＜ -- 乱数シードの固定 -- ＞
def setup_all_seed(seed=0):
    # numpyに関係する乱数シードの設定
    np.random.seed(seed)

    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

setup_all_seed()

# ＜ -- MNIST datasetの読み込み -- ＞
# 訓練データ
train_val_dataset = torchvision.datasets.MNIST(root='src/01_mnist/input',
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download = True)

# 検証データ
test_dataset = torchvision.datasets.MNIST(root='src/01_mnist/input',
                                        train=False,
                                        transform=transforms.ToTensor(),
                                        download = True)


# train : val = 80% : 20%
n_train = int(len(train_val_dataset) * 0.8)
n_val = len(train_val_dataset) - n_train

train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [n_train, n_val])


# print(len(train_dataset))
# print(type(train_dataset[0]))
# print(train_dataset[0][0].size())
# pprint(train_dataset[0][0])

# print(len(test_dataset))
# print(type(test_dataset[0]))
# print(test_dataset[0][0].size())
# pprint(test_dataset[0][0])

# fig, label = train_dataset[0]
# print("fig : {}, label : {}".format(fig,label))
# print("fig.size() : {}".format(fig.size()))


# plt.imshow(fig.view(-1,28), cmap='gray')
# plt.show()


# ＜ -- ミニバッチ学習の準備 -- ＞
batch_size = 256

# DataLoader：datasetからバッチごとに取り出すことを目的に使用
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                        batch_size=batch_size,
                                        shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                        batch_size=batch_size,
                                        shuffle=True)
# ＜ -- モデル定義 -- ＞

class Net(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(Net, self).__init__()

        # 使用する層の宣言
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)
        self.bn = nn.BatchNorm1d(input_size)

    def forward(self, x): # x : 入力
        x = self.bn(x)
        z1 = F.relu(self.fc1(x))
        z2 = F.relu(self.fc2(z1))
        y = self.fc3(z2)
        return y

input_size = 28*28
hidden1_size = 1024
hidden2_size = 512
output_size = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net(input_size, hidden1_size, hidden2_size, output_size).to(device)
print(model)

# ＜ -- 学習の準備 -- ＞

# 損失関数　criterion：基準
# CrossEntropyLoss：交差エントロピー誤差関数
criterion = nn.CrossEntropyLoss()

# 最適化法の指定　optimizer：最適化
# SGD：確率的勾配降下法
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 1epochの訓練を行う関数の定義
def train_model(model, train_loader, criterion, optimizer, device='cpu'):

    train_loss = 0.0
    num_train = 0

    # 学習モデルに変換
    model.train()

    for i, (images, labels) in enumerate(train_loader):
        # batch数をカウント
        num_train += len(labels)

        images, labels = images.view(-1, 28*28).to(device), labels.to(device)

        # 勾配を初期化
        optimizer.zero_grad()

        # 推論(順伝播)
        outputs = model(images)

        # 損失の算出
        loss = criterion(outputs, labels)

        # 誤差逆伝播
        loss.backward()

        # パラメータの更新
        optimizer.step()

        # lossを加算
        train_loss += loss.item()

    # lossの平均値を取る
    train_loss = train_loss / num_train

    return train_loss

# 検証データによるモデル評価を行う関数の定義
def test_model(model, test_loader, criterion, optimizer, device='cpu'):

    test_loss = 0.0
    num_test = 0

    # modelを評価モードに変更
    model.eval()

    with torch.no_grad(): # 勾配計算の無効化
        for i, (images, labels) in enumerate(test_loader):
            num_test += len(labels)
            images, labels = images.view(-1, 28*28).to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

        # lossの平均値を取る
        test_loss = test_loss / num_test
    return test_loss

# モデル学習を行う関数の定義

def lerning(model, train_loader, test_loader, criterion, optimizer, num_epochs, device='cpu'):

    train_loss_list = []
    test_loss_list = []

    # epoch数分繰り返す
    for epoch in range(1, num_epochs+1, 1):

        train_loss = train_model(model, train_loader, criterion, optimizer, device=device)
        test_loss = test_model(model, test_loader, criterion, optimizer, device=device)

        print("epoch : {}, train_loss : {:.5f}, test_loss : {:.5f}" .format(epoch, train_loss, test_loss))

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

    return train_loss_list, test_loss_list

# ＜ -- 学習 -- ＞
num_epochs = 10
train_loss_list, test_loss_list = lerning(model, train_loader, test_loader, criterion, optimizer, num_epochs, device=device)

# ＜ -- 学習済みモデルの保存 -- ＞
output_dir = "src/01_mnist/output/"
os.makedirs(output_dir, exist_ok=True)
torch.save(model, os.path.join(output_dir, 'mnist_origin.pth'))

# ＜ -- 学習推移のグラフ化 -- ＞
plt.plot(range(len(train_loss_list)), train_loss_list, c='b', label='train loss')
plt.plot(range(len(test_loss_list)), test_loss_list, c='r', label='test loss')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.grid()
plt.show()

# modelを評価モードに変更
model.eval()

with torch.no_grad(): # 勾配計算の無効化
    all_acc = 0
    print("##########################")
    for i, (x, t) in enumerate(val_loader):

        x = x.view(-1, 28*28).to(device)
        t = t.to(device)

        y = model(x)
        y_label = torch.argmax(y, dim=1)

        acc = torch.sum(y_label == t) * 1.0 / len(t)
        all_acc += acc

        # print("##########################")
        # print(x)
        # print(t)
        # print(y)
        # print(y_label)
        print(f'{i}: Accuracy: {acc * 100:.1f}%')

    print("##########################")
    print(f'{i}: MEAN Accuracy: {(all_acc / len(val_loader)) * 100:.1f}%')


