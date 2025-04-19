import torch
import numpy as np
import csv

import torch.nn.functional as F
import torch.nn
from torch.autograd import Variable
from pandas import read_csv
from scipy.stats import pearsonr
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import os
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error as mae
from scipy.stats import pearsonr

os.environ['KMP_DUPLICATE_LIB_OK']='True'
use_cuda = torch.cuda.is_available()

filename = "D:\计算机\脑与认知\lab\data/bl_info.csv"

def Myloader(path):
    return np.genfromtxt(path,delimiter=',')


# get a list of paths and labels.
def init_process(root_path):
    csv_reader = csv.reader(open(root_path))
    i = 0
    index = 0
    #data:最后返回值，每个元素是[filepath,label]
    data = []
    visited = []
    group = []
    for row in csv_reader:
        # 没有组别的人，不管
        if row[60] == '':
            continue
        if i > 0:
            # 已经访问过这个人的数据，就不再看他了
            if (row[0] in visited) == True:
                continue
            filepath = 'D:\计算机\脑与认知\lab\data\ex_data/' + row[2] + '_' + row[3] + '_ts.csv'
            #label = [0,0,1]
            label = 0
            if row[60]=='CN':
                label = [1,0,0]
                #label = 0
            elif row[60]=='MCI':
                label = [0,1,0]
                #label = 1
            else:
                label = [0,0,1]
                #label = 2
            new_data = Myloader(filepath)
            new_tensor = torch.FloatTensor(new_data.astype(np.float32))
            if torch.isnan(new_tensor).any()==True:
                continue
            visited.append(row[0])
            #data.append([filepath,label])
            data.append(new_data)
            group.append(label)
            index = index + 1
        i = i + 1
    return data,group



class load_data(torch.utils.data.Dataset):
    print('data processing...')

    def __init__(self, datapath = filename,mode='train'):
        self.datapath = datapath
        self.mode = mode

        data,label = init_process(datapath)
        train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.33)
        #train_data, val_data, test_data = data[:320], data[320:400], data[400:]
        #train_label, val_label, test_label = label[:320], label[320:400], label[400:]
        if self.mode == 'train':
            data = train_data
            label = train_label
        elif self.mode == 'test':
            data = test_data
            label = test_label
        else:
            data = train_data
            label = train_label
        self.data = torch.FloatTensor(np.expand_dims(data, 1).astype(np.float32))
        self.label = torch.FloatTensor(label)
        print(self.mode, self.data.shape, (self.label.shape))
    # shuffle
    #np.random.shuffle(data)
    # train, val, test = 320 + 80 + 130 = 530 60% 15% 25%
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = [self.data[idx], self.label[idx]]
        return sample


trainset = load_data(mode="train")
trainloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=False)




class E2EBlock(torch.nn.Module):
    # E2Eblock.

    def __init__(self, in_planes, planes, example, bias=False):
        super(E2EBlock, self).__init__()
        self.d = example.size(3)
        #self.d = dim
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a] * self.d, 3) + torch.cat([b] * self.d, 2)


class BrainNetCNN(torch.nn.Module):
    def __init__(self, example,num_classes=10):
        super(BrainNetCNN, self).__init__()
        self.in_planes = example.size(1)
        self.d = example.size(3)

        self.e2econv1 = E2EBlock(1, 32, example, bias=True)
        self.e2econv2 = E2EBlock(32, 64, example, bias=True)
        self.E2N = torch.nn.Conv2d(64, 1, (1, self.d))
        self.N2G = torch.nn.Conv2d(1, 256, (self.d, 1))
        self.dense1 = torch.nn.Linear(256, 128)
        self.dense2 = torch.nn.Linear(128, 30)
        self.dense3 = torch.nn.Linear(30, 3)

    def forward(self, x):
        out = F.leaky_relu(self.e2econv1(x), negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out), negative_slope=0.33)
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
        out = F.dropout(F.leaky_relu(self.N2G(out), negative_slope=0.33), p=0.5)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.leaky_relu(self.dense1(out), negative_slope=0.33), p=0.5)
        out = F.dropout(F.leaky_relu(self.dense2(out), negative_slope=0.33), p=0.5)
        out = F.leaky_relu(self.dense3(out), negative_slope=0.33)
        out = torch.nn.functional.softmax(out)

        return out

net = BrainNetCNN(trainset.data)

momentum = 0.9
lr = 0.0005
wd = 0.0005
#optimizer = torch.optim.SGD(net.parameters(),lr=lr,momentum=momentum,nesterov=True,weight_decay=wd)
optimizer = torch.optim.Adam(net.parameters(),lr=lr,betas=(0.9,0.99),weight_decay=wd)

def myloss(data,label):
    data_size = data.size(0)
    total_max = 0
    total_size = 0
    for i in range(data_size):
        if (torch.isnan(data[i]).any()==False):
            total_max = total_max + (data[i][0]-label[i][0])*(data[i][0]-label[i][0])\
                        + (data[i][1]-label[i][1])*(data[i][1]-label[i][1])\
                        + (data[i][2]-label[i][2])*(data[i][2]-label[i][2])
            total_size = total_size + 1
    if total_size == 0:
        return torch.tensor(1.0,requires_grad=True)
    else:
        return total_max/total_size


def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    running_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        if torch.isnan(inputs).any()==True:
            print(inputs)
            print('#################################')
        outputs = net(inputs)
        loss = myloss(outputs,targets)
        #loss = torch.nn.CrossEntropyLoss()(outputs,targets.long())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=10, norm_type=2)
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    return running_loss / batch_idx



def myaccuracy(data,label):
    print('data: ',data)
    print('label: ',label)
    data_size = data.size(0)
    total_correct = 0
    total_size = 0
    for i in range(data_size):
        if (torch.isnan(data[i]).any()==False):
            max_number,predict_index = torch.max(data[i],0)
            _,real_index = torch.max(label[i],0)
            #real_index = label[i]

            if predict_index == real_index :
                total_correct = total_correct + 1
            total_size = total_size + 1
    if total_size == 0:
        return -1.
    else:
        return total_correct,total_size


valset = load_data(mode="test")
valloader = torch.utils.data.DataLoader(valset, batch_size=15, shuffle=False)

#用validation测试
def test():
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    running_loss = 0.0
    runnung_accuracy = 0.0

    preds = []
    ytrue = []

    for batch_idx, (inputs, targets) in enumerate(valloader):

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)

            outputs = net(inputs)
            #loss = criterion(outputs, targets)
            #loss = myloss(outputs,targets)
            accuracy,size = myaccuracy(outputs, targets)

            #test_loss += loss.data[0]

            preds.append(outputs.numpy())
            ytrue.append(targets.numpy())

        # print statistics
            #running_loss += loss.item()
            correct += accuracy
            total += size

    #return np.vstack(preds), np.vstack(ytrue), running_loss / batch_idx
    return preds,ytrue, correct/total

nbepochs = 15
allacc_test = []
allloss_train = []



for epoch in range(nbepochs):
    loss_train = train(epoch)

    print(epoch,' loss: ',loss_train)

    preds, y_true, accuracy = test()

    print('accuracy: ',accuracy)

    allloss_train.append(loss_train)
    allacc_test.append(accuracy)

print(allloss_train)
print(allacc_test)
'''
plt.plot(np.linspace(0,len(allloss_test),len(allloss_test)),allloss_test)
plt.show()
plt.plot(np.linspace(0,len(allacc_test),len(allacc_test)),allacc_test)
plt.show()
'''
