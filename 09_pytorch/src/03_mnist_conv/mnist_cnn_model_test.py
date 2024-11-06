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

        # 使用する層の宣言
        self.conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)
        self.bn = nn.BatchNorm1d(input_size)

    def forward(self, x):

        x = self.conv(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = self.bn(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

model_path = "src/03_mnist_conv/output/mnist_origin.pth"
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

