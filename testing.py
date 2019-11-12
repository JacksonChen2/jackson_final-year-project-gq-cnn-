from main import BatchLoader
import numpy as np
import torch
import torch.nn
from torch import nn
import torchvision.transforms as transforms
import argparse
from Network import gqcnn
from  Network import improved_gqcnn

parser = argparse.ArgumentParser(description= \
                    'scipt for gqcnn')
parser.add_argument('--test_batchsize', default=40, type=int, help='trsing batches')
parser.add_argument('--output_path', default='/Users/jiazhengchen/'
                                             'Documents/Year4_S2/毕设/',
                    type=str, help='folder of output path')
args = parser.parse_args()

test_batchszie = args.test_batchsize

test_loader = BatchLoader(minibatches_path=args.output_path, normalize=True)

test_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

network = gqcnn()
if args.cuda:
    network = network.cuda()

def eval_net(net):
    net = net.eval()
    if cuda:
        net = net.cuda()

    net.load_state_dict(torch.load(args.output_path + 'gqcnn.pth', map_location='cpu'))

    correct = 0
    total = 0
    for batch_x, batch_y, batch_z in test_loader.load_test_batch(batch_size=test_batchszie, onehot=False):
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

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

if __name__ == '__main__':
# test your trained network
    eval_net(network)