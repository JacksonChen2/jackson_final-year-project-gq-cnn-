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

# parameter setting
parser = argparse.ArgumentParser(description= \
                    'scipt for gqcnn')
parser.add_argument('--train_epoch', default=10, type=int, help='number of train epoches')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--train_batchsize', default=100, type=int, help='training batches')
parser.add_argument('--test_batchsize', default=100, type=int, help='trsing batches')
parser.add_argument('--train_path', default='E:/Year4 S2/毕设/train_2', type=str, help='folder with train data')
parser.add_argument('--output_path', default='E:/Year4 S2/毕设/', type=str, help='folder of output path')
parser.add_argument('--test_path', default='E:/Year4 S2/毕设/test_2', type=str, help='folder with test data')
args = parser.parse_args()

train_batchsize = args.train_batchsize
test_batchszie = args.test_batchsize

# define the network
class gqcnn(nn.Module):
    def __init__(self):
        super(gqcnn, self).__init__()
        # root 1
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=7)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=5)
        self.lrn = nn.LocalResponseNorm(2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_3 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv1_4 = nn.Conv2d(64, 64, kernel_size=3)
        self.fc1_1 = nn.Linear(64 * 7 * 7, 1024)
        #root 2
        self.fc2_1 = nn.Linear(1, 16)
        #root 3
        self.fc3_1 = nn.Linear(1040, 1024)
        self.fc3_2 = nn.Linear(1024, 2)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()



    def forward(self, x1, x2):
        x1 = F.relu(self.conv1_1(x1))
        x1 = self.pool(self.lrn(F.relu(self.conv1_2(x1))))
        x1 = F.relu(self.conv1_3(x1))
        x1 = self.lrn(F.relu(x1))
        x1 = self.lrn(F.relu(self.conv1_4(x1)))
        x1 = x1.view(-1, 64 * 7 * 7)
        #print(x1.shape)
        x1 = self.fc1_1(x1)
        x1 = F.relu(x1)
        #print(x1.shape)
        x2 = x2.view(-1, 1)
        #print(x2.shape)
        x2 = F.relu(self.fc2_1(x2))
        x3 = torch.cat((x1, x2), dim=1)
        x3 = F.relu(self.fc3_1(x3))
        x3 = self.softmax(self.fc3_2(x3))
        #print(x3.shape)
        return x3

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
                # grasp = dct['grasp'].astype(np.float32)
                # depth = dct['depth'].astype(np.float32)
                grasp = dct['grasp'] / 255
                depth = dct['depth']
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

model = gqcnn()
if cuda:
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(), lr=args.lr)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
#scheduler = optim.lr_scheduler.ExponentialLR(optimizer,0.85) # define my own lr schedulers

# get train and test dataset
train_batch_loader = BatchLoader(minibatches_path=args.train_path, normalize=True)
test_batch_loader = BatchLoader(minibatches_path=args.test_path, normalize=True)
number_of_image = 0
for batch_x, batch_y, batch_z in train_batch_loader.load_train_batch(batch_size=train_batchsize, onehot=False):
    number_of_image += batch_x.shape[0]
iteration = int(number_of_image/train_batchsize)
#print(iteration)

#define training function
def train_net(net, criterion, optimizer, epochs=args.train_epoch):
    print('start training')
    net = net.train()
    train_loss_list = []
    for epoch in range(epochs):

        running_loss = 0.0
        #running_time =0.0
        i = 0

        for batch_x, batch_y, batch_z in train_batch_loader.load_train_batch(batch_size=train_batchsize,
                                                                             onehot=False):
            #start_time = time.time()
            # change the sequence of different dimension
            batch_x = np.transpose(batch_x, (0, 3, 1, 2))
            # change it from [batch size,] to [1, batch size]
            batch_y = batch_y.reshape(-1, 1)
            # transfer from Byte type to long type
            batch_z = batch_z.astype(int)
            #print(type(batch_z))
            grasp, depth, labels = torch.from_numpy(batch_x), torch.from_numpy(batch_y), torch.from_numpy(batch_z)
            labels = labels.long()
            grasp = grasp.float()
            if cuda:
                grasp, depth, labels = grasp.cuda(), depth.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = net(grasp, depth)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if cuda:
                loss = loss.cpu()

            running_loss += loss.item()
            i += 1
            #elapsed_time = time.time() - start_time
            #running_time += elapsed_time.item()
            if i % 2000 == 1999:
                print('[%d, %5d] train loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                #print(grasp)
                train_loss_list.append(running_loss / 2000)
                running_loss = 0.0
                running_time = 0.0

        # if type(scheduler).__name__ != 'NoneType':
        #     scheduler.step()

    torch.save(net.state_dict(), args.output_path + 'gqcnn.pth')
    print('finish training')

def eval_net(net):
    net = net.eval()
    if cuda:
        net = net.cuda()

    net.load_state_dict(torch.load(args.output_path + 'gqcnn.pth', map_location='cpu'))

    correct = 0
    total = 0
    for batch_x, batch_y, batch_z in test_batch_loader.load_test_batch(batch_size=test_batchszie, onehot=False):
        # change the sequence of different dimension
        batch_x = np.transpose(batch_x, (0, 3, 1, 2))
        # change it from [batch size,] to [1, batch size]
        batch_y = batch_y.reshape(-1,1)
        # transfer from Byte type to long type
        batch_z = batch_z.astype(int)
        #print(type(batch_z))
        grasp, depth, labels = torch.from_numpy(batch_x), torch.from_numpy(batch_y), torch.from_numpy(batch_z)
        labels = labels.long()
        grasp = grasp.float()
        if cuda:
            grasp, depth, labels = grasp.cuda(), depth.cuda(), labels.cuda()
        # print(grasp)
        #         # print(depth)
        outputs = net(grasp, depth)

        if cuda:
            outputs = outputs.cpu()
            labels = labels.cpu()
        print(outputs)

        # for i in range(test_batchszie):
        #     if outputs[i][1] > 0.02:
        #         outputs[i][1] = 1
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the all test images: %d %%' % (
            100 * correct / total))



# run the
if __name__ == '__main__':     # this is used for running in Windows
    # train modified network
    train_net(model, criterion, optimizer)
    eval_net(model)


    # test the baseline network and modified network
    #eval_net(baseline, testloader, logging, mode="baseline")
    #eval_net(modified, testloader, logging, mode="modified")








