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
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)

    def forward(self, x): # x : 入力
        z1 = F.relu(self.fc1(x))
        z2 = F.relu(self.fc2(z1))
        y = self.fc3(z2)
        return y

model_path = "src/01_mnist/output/mnist_origin.pth"
model = torch.load(model_path)
model.eval()

# 検証データ
test_dataset = torchvision.datasets.MNIST(root='src/01_mnist/input',
                                        train=False,
                                        transform=transforms.ToTensor(),
                                        download = True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

plt.figure(figsize=(20, 10))
for i in range(10):
    image, label = test_dataset[i]
    image = image.view(-1, 28*28).to(device)

    print("###########################")
    print(label)

    # 推論
    estim = model(image)
    print(estim)
    prediction_label = torch.argmax(estim)
    print(prediction_label)
    print()


    ax = plt.subplot(1, 10, i+1)

    plt.imshow(image.detach().to('cpu').numpy().reshape(28, 28), cmap='gray')
    ax.axis('off')
    ax.set_title('label : {}\n Prediction : {}'.format(label, prediction_label), fontsize=15)
plt.show()

