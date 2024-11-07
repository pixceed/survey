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
train_val_dataset = torchvision.datasets.MNIST(root='./data',
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download = True)

# 検証データ
test_dataset = torchvision.datasets.MNIST(root='./data',
                                        train=False,
                                        transform=transforms.ToTensor(),
                                        download = True)

print(type(train_val_dataset))
print(train_val_dataset)