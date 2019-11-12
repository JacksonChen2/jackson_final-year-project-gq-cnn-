import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
import torchvision.transforms as transforms
import argparse
import os
import time
from Network import gqcnn
from  Network import improved_gqcnn

# parameter setting
parser = argparse.ArgumentParser(description= \
                    'scipt for gqcnn')
parser.add_argument('--train_epoch', default=10, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--train_batchsize', default=20, type=int, help='training batches')
parser.add_argument('--validation_batchsize', default=20, type=int, help='validation batches')
parser.add_argument('--test_batchsize', default=40, type=int, help='trsing batches')
parser.add_argument('--train_path', default='/Users/jiazhengchen/Documents/'
                                            'Year4_S2/dextnet_data/train',
                    type=str, help='folder with train data')
parser.add_argument('--val_path', default='/Users/jiazhengchen/Documents/'
                                          'Year4_S2/dextnet_data/val',
                    type=str, help='folder with validation data')
parser.add_argument('--output_path', default='/Users/jiazhengchen/'
                                             'Documents/Year4_S2/毕设/',
                    type=str, help='folder of output path')
args = parser.parse_args()

train_batchsize = args.train_batchsize
val_batchsize = args.validation_batchsize
test_batchszie = args.test_batchsize


# function to get the minibatch from npz file
class BatchLoader():
    def __init__(self, minibatches_path,
                 normalize=True):

        self.normalize = normalize

        self.train_batchpaths = [os.path.join(minibatches_path, f)
                                 for f in os.listdir(minibatches_path)
                                 if f.startswith('train')]
        self.valid_batchpaths = [os.path.join(minibatches_path, f)
                                 for f in os.listdir(minibatches_path)
                                 if f.startswith('valid')]
        self.test_batchpaths = [os.path.join(minibatches_path, f)
                                for f in os.listdir(minibatches_path)
                                if f.startswith('train')]

        self.n_classes = 2


    def load_train_batch(self, batch_size, onehot=False,
                         shuffle_within=False, shuffle_paths=False,
                         seed=None):
        for batch_x, batch_y, batch_z in self._load_batch(which='train',
                                                 batch_size=batch_size,
                                                 onehot=onehot,
                                                 shuffle_within=shuffle_within,
                                                 shuffle_paths=shuffle_paths,
                                                 seed=seed):
            yield batch_x, batch_y, batch_z

    def load_test_batch(self, batch_size, onehot=False,
                        shuffle_within=False, shuffle_paths=False,
                        seed=None):
        for batch_x, batch_y, batch_z in self._load_batch(which='test',
                                                 batch_size=batch_size,
                                                 onehot=onehot,
                                                 shuffle_within=shuffle_within,
                                                 shuffle_paths=shuffle_paths,
                                                 seed=seed):
            yield batch_x, batch_y, batch_z

    def load_validation_batch(self, batch_size, onehot=False,
                              shuffle_within=False, shuffle_paths=False,
                              seed=None):
        for batch_x, batch_y, batch_z in self._load_batch(which='valid',
                                                 batch_size=batch_size,
                                                 onehot=onehot,
                                                 shuffle_within=shuffle_within,
                                                 shuffle_paths=shuffle_paths,
                                                 seed=seed):
            yield batch_x, batch_y, batch_z

    def _load_batch(self, which='train', batch_size=50, onehot=False,
                    shuffle_within=True, shuffle_paths=True, seed=None):

        if which == 'train':
            paths = self.train_batchpaths
        elif which == 'valid':
            paths = self.valid_batchpaths
        elif which == 'test':
            paths = self.test_batchpaths
        else:
            raise ValueError('`which` must be "train" or "test". Got %s.' %
                             which)

        rgen = np.random.RandomState(seed)
        if shuffle_paths:
            paths = rgen.shuffle(paths)

        for batch in paths:

            dct = np.load(batch)

            if onehot:
                labels = (np.arange(self.n_classes) ==
                          dct['label'][:, None]).astype(np.uint8)
            else:
                labels = dct['label']


            if self.normalize:
                # normalize to [0, 1] range
                grasp = dct['grasp'].astype(np.float32)/255
                depth = dct['depth'].astype(np.float32)
            else:
                grasp = dct['grasp']
                depth = dct['depth']

            arrays = [grasp, depth, labels]
            del dct
            indices = np.arange(arrays[0].shape[0])

            if shuffle_within:
                rgen.shuffle(indices)

            for start_idx in range(0, indices.shape[0] - batch_size + 1,
                                   batch_size):
                index_slice = indices[start_idx:start_idx + batch_size]
                yield (ary[index_slice] for ary in arrays)

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = improved_gqcnn()
if cuda:
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [8,16,24,30], gamma=0.1)

# get train and test dataset
batch_loader = BatchLoader(minibatches_path=args.train_path, normalize=True)
val_loader = BatchLoader(minibatches_path=args.val_path, normalize=True)
number_of_image = 0
for batch_x, batch_y, batch_z in batch_loader.load_train_batch(batch_size=train_batchsize, onehot=True):
    number_of_image += batch_x.shape[0]
iteration = int(number_of_image/train_batchsize)
#print(iteration)

# trasform function
train_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

test_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

#define training function
def train_net(net, criterion, optimizer, scheduler, epochs=args.train_epoch):
    net = net.train()
    train_loss_list = []
    for epoch in range(epochs):
        running_loss = 0.0
        i = 0

        for batch_x, batch_y, batch_z in batch_loader.load_train_batch(batch_size=train_batchsize, onehot=False):
            # change the sequence of different dimension
            batch_x = np.transpose(batch_x, (0,3,1,2))
            # change it from [batch size,] to [1, batch size]
            batch_y = batch_y.reshape(-1,1)
            # transfer from Byte type to long type
            batch_z = batch_z.astype(int)
            #print(type(batch_z))
            grasp, depth, labels = torch.from_numpy(batch_x), torch.from_numpy(batch_y), torch.from_numpy(batch_z)
            if cuda:
                grasp, labels = grasp.cuda(), depth.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = net(grasp, depth)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if cuda:
                loss = loss.cpu()

            running_loss += loss.item()
            i += 1
            if i % 100 == 99:
                print('[%d, %5d] train loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                train_loss_list.append(running_loss/100)
                running_loss = 0.0

            if type(scheduler).__name__ != 'NoneType':
                scheduler.step()

            eval_net(net)

            torch.save(net.state_dict(), args.output_path + 'gqcnn_%d.pth' % (epoch + 1))

    torch.save(net.state_dict(), args.output_path + 'gqcnn.pth')
    print('finish training')

def eval_net(net):
    net = net.eval()
    if cuda:
        net = net.cuda()

    net.load_state_dict(torch.load(args.output_path + 'gqcnn.pth', map_location='cpu'))

    correct = 0
    total = 0
    for batch_x, batch_y, batch_z in batch_loader.load_test_batch(batch_size=val_batchsize, onehot=False):
        # change the sequence of different dimension
        batch_x = np.transpose(batch_x, (0,3,1,2))
        # change it from [batch size,] to [1, batch size]
        batch_y = batch_y.reshape(-1,1)
        # transfer from Byte type to long type
        batch_z = batch_z.astype(int)
        #print(type(batch_z))
        grasp, depth, labels = torch.from_numpy(batch_x), torch.from_numpy(batch_y), torch.from_numpy(batch_z)
        if cuda:
            grasp, labels = grasp.cuda(), depth.cuda(), labels.cuda()
        outputs = net(grasp, depth)

        if cuda:
            outputs = outputs.cpu()
            labels = labels.cpu()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 validation images: %d %%' % (
            100 * correct / total))

# run the
if __name__ == '__main__':     # this is used for running in Windows
    # train modified network
    train_net(model, criterion, optimizer, scheduler)
