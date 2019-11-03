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


class MobileNet_v2(torch.nn.Module): 
    def __init__(self):
        
        # define block
        class MobileNetBlock(torch.nn.Module):
            def __init__(self, in_dim, out_dim, stride=1, expansion_t=6, split_division_by=8):
                super(MobileNetBlock, self).__init__()
                
                self.module = torch.nn.Sequential(
                    torch.nn.Conv2d(in_dim, in_dim * expansion_t, kernel_size=1, padding=0, stride=1, groups=in_dim),
                    torch.nn.BatchNorm2d(in_dim * expansion_t),
                    torch.nn.ReLU6(),
                    torch.nn.Conv2d(in_dim * expansion_t, in_dim * expansion_t, kernel_size=3, padding=1, stride=stride, groups=split_division_by),
                    torch.nn.BatchNorm2d(in_dim * expansion_t),
                    torch.nn.ReLU6(),
                    torch.nn.Conv2d(in_dim * expansion_t, out_dim, kernel_size=1, padding=0, stride=1),
                    torch.nn.BatchNorm2d(out_dim),
                )
                    
            def forward(self, _input):
                x = self.module(_input)
                
                # if shape matches, add skip connection
                if x.size() == _input.size():
                    x = x + _input
                
                return x
            
            
        # define feature dimension flattening layer
        class Flatten(torch.nn.Module):
            def forward(self, x):
                x = x.view(x.size()[0], -1)
                return x
        
        
        super(MobileNet_v2, self).__init__()
        
        self.module = torch.nn.Sequential(
            # input
            # 224 x 224 x 3
            torch.nn.Conv2d(channel, 32, kernel_size=3, padding=1, stride=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU6(),
            # 112 x 112 x 32
            MobileNetBlock(32, 16, expansion_t=1),
            # 112 x 112 x 16
            MobileNetBlock(16, 24, stride=2),
            MobileNetBlock(24, 24),
            # 56 x 56 x 24
            MobileNetBlock(24, 32, stride=2),
            MobileNetBlock(32, 32),
            MobileNetBlock(32, 32),
            # 28 x 28 x 32
            MobileNetBlock(32, 64, stride=2),
            MobileNetBlock(64, 64),
            MobileNetBlock(64, 64),
            MobileNetBlock(64, 64),
            # 14 x 14 x 64
            MobileNetBlock(64, 96),
            MobileNetBlock(96, 96),
            MobileNetBlock(96, 96),
            # 14 x 14 x 96
            MobileNetBlock(96, 160, stride=2),
            MobileNetBlock(160, 160),
            MobileNetBlock(160, 160),
            # 7 x 7 x 160
            MobileNetBlock(160, 320),
            # 7 x 7 x 320
            torch.nn.Conv2d(320, 1280, kernel_size=1, padding=0, stride=1),
            torch.nn.BatchNorm2d(1280),
            torch.nn.ReLU6(),
            # 7 x 7 x 1280
            torch.nn.AdaptiveAvgPool2d([1,1]),
            Flatten(),
            # 1 x 1 x 1280
            torch.nn.Linear(1280, class_N),
            torch.nn.Softmax(dim=1)
        )

        
    def forward(self, x):
        x = self.module(x)
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
    model = MobileNet_v1().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()

    xs, ts, paths = data_load('../Dataset/train/images/', hf=True, vf=True, rot=10)

    # training
    mb = 32
    mbi = 0
    train_ind = np.arange(len(xs))
    np.random.seed(0)
    np.random.shuffle(train_ind)

    loss_fn = torch.nn.CNLLLoss()
    
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
    model = MobileNet_v1().to(device)
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
