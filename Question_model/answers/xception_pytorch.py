import torch
import torch.nn.functional as F
import argparse
import cv2
import numpy as np
from glob import glob
import copy

num_classes = 2
img_height, img_width = 96, 96
channel = 3
GPU = False
torch.manual_seed(0)


class Block(torch.nn.Module):
    def __init__(self, dim=728, cardinality=1):
        super(Block, self).__init__()

        self.block = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=cardinality),
            torch.nn.BatchNorm2d(dim),
            torch.nn.ReLU(),
            torch.nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=cardinality),
            torch.nn.BatchNorm2d(dim),
            torch.nn.ReLU(),
            torch.nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=cardinality),
            torch.nn.BatchNorm2d(dim),
        )
        
    def forward(self, x):
        res_x = self.block(x)            
        x = torch.add(res_x, x)

        return x

        

class Xception(torch.nn.Module):
    def __init__(self):
        super(Xception, self).__init__()

        # Entry flow
        self.conv1 = torch.nn.Conv2d(channel, 32, kernel_size=3, padding=1, stride=2)
        self.bn1 = torch.nn.BatchNorm2d(32)
        
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, groups=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, groups=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(3, stride=2, padding=1))
        self.conv3_sc = torch.nn.Conv2d(64, 128, kernel_size=1, padding=0, stride=2)
        self.bn3_sc = torch.nn.BatchNorm2d(128)
        
        self.conv4 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, groups=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, groups=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.MaxPool2d(3, stride=2, padding=1))
        self.conv4_sc = torch.nn.Conv2d(128, 256, kernel_size=1, padding=0, stride=2)
        self.bn4_sc = torch.nn.BatchNorm2d(256)
        
        self.conv5 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 728, kernel_size=3, padding=1, stride=1, groups=1),
            torch.nn.BatchNorm2d(728),
            torch.nn.ReLU(),
            torch.nn.Conv2d(728, 728, kernel_size=3, padding=1, stride=1, groups=1),
            torch.nn.BatchNorm2d(728),
            torch.nn.MaxPool2d(3, stride=2, padding=1))
        self.conv5_sc = torch.nn.Conv2d(256, 728, kernel_size=1, padding=0, stride=2)
        self.bn5_sc = torch.nn.BatchNorm2d(728)
        
        # Middle flow
        self.middle_flow = torch.nn.Sequential(
            *[Block() for _ in range(8)]
        )
        
        # Exit flow
        self.conv_exit1 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(728, 728, kernel_size=3, padding=1, stride=1, groups=1),
            torch.nn.BatchNorm2d(728),
            torch.nn.ReLU(),
            torch.nn.Conv2d(728, 1024, kernel_size=3, padding=1, stride=1, groups=1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.MaxPool2d(3, stride=2, padding=1))
        self.conv_exit1_sc = torch.nn.Conv2d(728, 1024, kernel_size=1, padding=0, stride=2)
        self.bn_exit1_sc = torch.nn.BatchNorm2d(1024)
        
        self.conv_exit2 = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 1536, kernel_size=3, padding=1, stride=1, groups=1),
            torch.nn.BatchNorm2d(1536),
            torch.nn.ReLU(),
            torch.nn.Conv2d(1536, 2048, kernel_size=3, padding=1, stride=1, groups=1),
            torch.nn.BatchNorm2d(2048),)
        
        self.linear = torch.nn.Linear(2048, num_classes)
        
        
    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x_sc = self.conv3_sc(x)
        x_sc = self.bn3_sc(x_sc)
        x = self.conv3(x)
        x = torch.add(x_sc, x)
        
        x_sc = self.conv4_sc(x_sc)
        x_sc = self.bn4_sc(x_sc)
        x = self.conv4(x)
        x = torch.add(x_sc, x)
        
        x_sc = self.conv5_sc(x_sc)
        x_sc = self.bn5_sc(x_sc)
        x = self.conv5(x)
        x = torch.add(x_sc, x)
        
        # Middle flow
        x = self.middle_flow(x)
        
        # Exit flow
        x_sc = self.conv_exit1_sc(x)
        x_sc = self.bn_exit1_sc(x_sc)
        x = self.conv_exit1(x)
        x = torch.add(x_sc, x)
        
        x = self.conv_exit2(x)

        x = F.avg_pool2d(x, [img_height // 32, img_width // 32], padding=0, stride=1)
        x = x.view(list(x.size())[0], -1)
        x = self.linear(x)
        x = F.softmax(x, dim=1)
        
        return x



    
CLS = ['akahara', 'madara']

# get train data
def data_load(path, hf=False, vf=False, rot=False):
    xs = []
    ts = []
    paths = []
    
    for dir_path in glob(path + '/*'):
        for path in glob(dir_path + '/*'):
            x = cv2.imread(path)
            x = cv2.resize(x, (img_width, img_height)).astype(np.float32)
            x /= 255.
            x = x[..., ::-1]
            xs.append(x)

            for i, cls in enumerate(CLS):
                if cls in path:
                    t = i
            
            ts.append(t)

            paths.append(path)

            if hf:
                xs.append(x[:, ::-1])
                ts.append(t)
                paths.append(path)

            if vf:
                xs.append(x[::-1])
                ts.append(t)
                paths.append(path)

            if hf and vf:
                xs.append(x[::-1, ::-1])
                ts.append(t)
                paths.append(path)

            if rot != False:
                angle = rot
                scale = 1

                # show
                a_num = 360 // rot
                w_num = np.ceil(np.sqrt(a_num))
                h_num = np.ceil(a_num / w_num)
                count = 1
                #plt.subplot(h_num, w_num, count)
                #plt.axis('off')
                #plt.imshow(x)
                #plt.title("angle=0")
                
                while angle < 360:
                    _h, _w, _c = x.shape
                    max_side = max(_h, _w)
                    tmp = np.zeros((max_side, max_side, _c))
                    tx = int((max_side - _w) / 2)
                    ty = int((max_side - _h) / 2)
                    tmp[ty: ty+_h, tx: tx+_w] = x.copy()
                    M = cv2.getRotationMatrix2D((max_side/2, max_side/2), angle, scale)
                    _x = cv2.warpAffine(tmp, M, (max_side, max_side))
                    _x = _x[tx:tx+_w, ty:ty+_h]
                    xs.append(_x)
                    ts.append(t)
                    paths.append(path)

                    # show
                    #count += 1
                    #plt.subplot(h_num, w_num, count)
                    #plt.imshow(_x)
                    #plt.axis('off')
                    #plt.title("angle={}".format(angle))

                    angle += rot
                #plt.show()


    xs = np.array(xs, dtype=np.float32)
    ts = np.array(ts, dtype=np.int)
    
    xs = xs.transpose(0,3,1,2)

    return xs, ts, paths



# train
def train():
    # GPU
    device = torch.device("cuda" if GPU else "cpu")

    # model
    model = Xception().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()

    xs, ts, paths = data_load('../Dataset/train/images/', hf=True, vf=True, rot=10)

    # training
    mb = 32
    mbi = 0
    train_ind = np.arange(len(xs))
    np.random.seed(0)
    np.random.shuffle(train_ind)

    loss_fn = torch.nn.NLLLoss()
    
    for i in range(500):
        if mbi + mb > len(xs):
            mb_ind = copy.copy(train_ind)[mbi:]
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb-(len(xs)-mbi))]))
        else:
            mb_ind = train_ind[mbi: mbi+mb]
            mbi += mb

        x = torch.tensor(xs[mb_ind], dtype=torch.float).to(device)
        t = torch.tensor(ts[mb_ind], dtype=torch.long).to(device)

        opt.zero_grad()
        y = model(x)
        #y = F.log_softmax(y, dim=1)
        loss = loss_fn(torch.log(y), t)
        
        loss.backward()
        opt.step()
    
        pred = y.argmax(dim=1, keepdim=True)
        acc = pred.eq(t.view_as(pred)).sum().item() / mb

        if (i + 1) % 50 == 0:
            print("iter >>", i+1, ', loss >>', loss.item(), ', accuracy >>', acc)

    torch.save(model.state_dict(), 'cnn.pt')

# test
def test():
    device = torch.device("cuda" if GPU else "cpu")
    model = Xception().to(device)
    model.eval()
    model.load_state_dict(torch.load('cnn.pt'))

    xs, ts, paths = data_load('../Dataset/test/images/')

    for i in range(len(paths)):
        x = xs[i]
        t = ts[i]
        path = paths[i]
        
        x = np.expand_dims(x, axis=0)
        x = torch.tensor(x, dtype=torch.float).to(device)
        
        pred = model(x)
        pred = F.softmax(pred, dim=1).detach().cpu().numpy()[0]
    
        print("in {}, predicted probabilities >> {}".format(path, pred))
    

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
