import torch
from PIL import Image
import torch.nn as nn
import torchvision
from torchvision import models
import torchvision.transforms as transforms
import torch.utils.data.dataloader as DataLoader
import torch.utils.data.dataset as Dataset
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import os
import torchsnooper


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)
torch.backends.cudnn.benchmark = True

train_info_path = './dataset/train/annotation.txt'
test_info_path = './dataset/test/annotation.txt'
train_data_path = './dataset/train'
test_data_path = './dataset/test'
dic = {'#ffff00': 0, '#000000': 1, '#ffffff': 2, '#0000ff': 3, '#00ffff': 4, '#ff0000': 5, '#00ff00': 6}
out_channels = 7
epochs = 200
batch_sizes = 16
learning_rate = 0.001
pre_trained = False


class Sdataset(Dataset.Dataset):
    def __init__(self, info, dir, transform):
        super().__init__()
        self.df = info
        self.data_dir = dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idex):
        img_name, label = self.df[idex]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path)
        image = self.transform(image)
        img = torch.Tensor(image)
        # lab = torch.zeros(out_channels)
        # lab[dic[label]] = 1
        lab = dic[label]
        return img, lab


class Net(nn.Module):
    def __init__(self, base_model):
        super(Net, self).__init__()
        # mid_num = base_model.fc.in_features
        self.resnet_layer = nn.Sequential(*list(base_model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, out_channels),
        )
        # self.Linear_layer = nn.Linear(mid_num, out_channels)

    def forward(self, x):
        x = self.resnet_layer(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # x = self.Linear_layer(x)
        # x = torch.nn.functional.softmax(x, 1)
        return x


train_info = []
test_info = []
color = []
with open(train_info_path) as f:
    for line in f.readlines():
        train_info.append(tuple(line.strip().split(' ')))
with open(test_info_path) as f:
    for line in f.readlines():
        test_info.append(tuple(line.strip().split(' ')))

transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     # transforms.RandomCrop((224, 224)),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize(
         mean=(0.5, 0.5, 0.5),
         std=(0.5, 0.5, 0.5)
     )])

trainset = Sdataset(train_info, train_data_path, transform=transform)
testset = Sdataset(test_info, test_data_path, transform=transform)


trainloader = DataLoader.DataLoader(dataset=trainset, batch_size=batch_sizes, shuffle=True, num_workers=0)
testloader = DataLoader.DataLoader(dataset=trainset, batch_size=batch_sizes, shuffle=True, num_workers=0)

if not pre_trained:
    base_model = models.vgg16(pretrained=False)
    # base_model = models.resnet50(pretrained=False)
    model = Net(base_model)
else:
    model = torch.load("./model/trained.pth")
# criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),  lr=learning_rate, momentum=0.9)

model = model.to(device)
print(model)

loss_draw = []
def train():
    for epoch in range(epochs):

        for i, data in enumerate(trainloader):
            # get the input
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.int64)
            # plt.imshow(inputs[0].cpu().numpy().transpose((1, 2, 0)))
            # plt.show()
            # break

            optimizer.zero_grad()

            outputs = model(inputs)
            # print(outputs, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))  # 每2000次迭代，输出loss的平均值
            loss_draw.append(loss.item())

        plt.plot(loss_draw)
        plt.savefig('./model/loss.png')
        plt.clf()

        correct = 0
        all = 0
        for i, data in enumerate(testloader):
            # get the input
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.int64)
            # plt.imshow(inputs[0].cpu().numpy().transpose((1, 2, 0)))
            # plt.show()
            # break

            outputs = model(inputs)
            pred = outputs.max(1)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()
            all += batch_sizes
        print("correct rate", correct/all)

    print('Finished Training')
    torch.save(model.cpu(), './model/trained.pth')

train()