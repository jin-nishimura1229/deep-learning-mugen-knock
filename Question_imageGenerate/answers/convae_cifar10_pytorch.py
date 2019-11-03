import torch
import torch.nn.functional as F
import argparse
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

num_classes = 2
img_height, img_width = 32, 32
channel = 3

GPU = True
torch.manual_seed(0)
    
class Mynet(torch.nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()

        self.enc1 = torch.nn.Conv2d(channel, 32, kernel_size=3, padding=1)
        self.enc2 = torch.nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.dec2 = torch.nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2)
        self.dec1 = torch.nn.ConvTranspose2d(32, channel, kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.enc1(x)
        x = F.max_pool2d(x, 2)
        x = self.enc2(x)
        x = F.max_pool2d(x, 2)
        x = self.dec2(x)
        x = self.dec1(x)
        return x

    
import pickle
import os
    
def load_cifar10():

    path = 'cifar-10-batches-py'

    if not os.path.exists(path):
        os.system("wget {}".format(path))
        os.system("tar xvf {}".format(path))

    # train data
    
    train_x = np.ndarray([0, 32, 32, 3], dtype=np.float32)
    train_y = np.ndarray([0, ], dtype=np.int)
    
    for i in range(1, 6):
        data_path = path + '/data_batch_{}'.format(i)
        with open(data_path, 'rb') as f:
            datas = pickle.load(f, encoding='bytes')
            print(data_path)
            x = datas[b'data']
            x = x.reshape(x.shape[0], 3, 32, 32)
            x = x.transpose(0, 2, 3, 1)
            train_x = np.vstack((train_x, x))
        
            y = np.array(datas[b'labels'], dtype=np.int)
            train_y = np.hstack((train_y, y))

    # test data
    
    data_path = path + '/test_batch'
    
    with open(data_path, 'rb') as f:
        datas = pickle.load(f, encoding='bytes')
        print(data_path)
        x = datas[b'data']
        x = x.reshape(x.shape[0], 3, 32, 32)
        test_x = x.transpose(0, 2, 3, 1)
    
        test_y = np.array(datas[b'labels'], dtype=np.int)

    return train_x, train_y, test_x, test_y


# train
def train():
    # GPU
    device = torch.device("cuda" if GPU else "cpu")

    # model
    model = Mynet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    train_x, train_y, test_x, test_y = load_cifar10()
    xs = train_x / 255
    xs = xs.transpose(0, 3, 1, 2)

    # training
    mb = 512
    mbi = 0
    train_ind = np.arange(len(xs))
    np.random.seed(0)
    np.random.shuffle(train_ind)
    
    for i in range(5000):
        if mbi + mb > len(xs):
            mb_ind = train_ind[mbi:]
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb-(len(xs)-mbi))]))
            mbi = mb - (len(xs) - mbi)
        else:
            mb_ind = train_ind[mbi: mbi+mb]
            mbi += mb

        x = torch.tensor(xs[mb_ind], dtype=torch.float).to(device)
        t = torch.tensor(xs[mb_ind], dtype=torch.float).to(device)

        opt.zero_grad()

        y = model(x)
        loss = torch.nn.MSELoss()(y, t)
        loss.backward()
        opt.step()
    
        #pred = y.argmax(dim=1, keepdim=True)
        acc = y.eq(t.view_as(y)).sum().item() / mb

        if (i+1) % 100 == 0:
            print("iter >>", i+1, ',loss >>', loss.item(), ',accuracy >>', acc)

    torch.save(model.state_dict(), 'cnn.pt')

# test
def test():
    device = torch.device("cuda" if GPU else "cpu")
    model = Mynet().to(device)
    model.eval()
    model.load_state_dict(torch.load('cnn.pt'))

    train_x, train_y, test_x, test_y = load_cifar10()
    xs = test_x / 255
    xs = xs.transpose(0, 3, 1, 2)

    for i in range(10):
        x = xs[i]
        
        x = np.expand_dims(x, axis=0)
        x = torch.tensor(x, dtype=torch.float).to(device)
        
        pred = model(x)

        pred = pred.view(channel, img_height, img_width)
        pred = pred.detach().cpu().numpy()
        pred -= pred.min()
        pred /= pred.max()
        pred = pred.transpose(1,2,0)

        
        _x = x.detach().cpu().numpy()[0]
        #_x = (_x + 1) / 2
        if channel == 1:
            pred = pred[..., 0]
            _x = _x[0]
            cmap = 'gray'
        else:
            _x = _x.transpose(1,2,0)
            cmap = None

        plt.subplot(1,2,1)
        plt.title("input")
        plt.imshow(_x, cmap=cmap)
        plt.subplot(1,2,2)
        plt.title("predicted")
        plt.imshow(pred, cmap=cmap)
        plt.show()
    

def arg_parse():
    parser = argparse.ArgumentParser(description='CNN implemented with Keras')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')
    args = parser.parse_args()
    return args

# main
if __name__ == '__main__':
    args = arg_parse()

    if args.train:
        train()
    if args.test:
        test()

    if not (args.train or args.test):
        print("please select train or test flag")
        print("train: python main.py --train")
        print("test:  python main.py --test")
        print("both:  python main.py --train --test")
