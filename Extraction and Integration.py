import numpy as np
#
for o in range(0,6):
    for k in range(0,10):
        for j in range(0,10):
            for i in range(0,10):
                a = np.load('E:/Year4 S2/毕设/grasp_255/grasp_255_0%d%d%d%d.npz' % (o,k,j,i))['grasp']
                b = np.load('E:/Year4 S2/毕设/depth/depth_0%d%d%d%d.npz' %(o,k,j,i))['arr_0']
                c = np.load('E:/Year4 S2/毕设/label/grasp_labels_0%d%d%d%d.npz' %(o,k,j,i))['arr_0']
                np.savez("train_0%d%d%d%d.npz" %(o,k,j,i), grasp = a, depth = b, label = c)

a = np.load('F:/Year4 S1/ELEC5306/project2/final/train_00000.npz')['grasp']
for o in range(6,7):
    for k in range(0,7):
        for j in range(0,10):
            for i in range(0,10):
                depth = []
                for p in range(0,1000):
                    a = np.load('F:/dexnet_data/hand_configuration/hand_configurations_0%d%d%d%d.npz'
                                %(o,k,j,i))['arr_0'][p][2]
                    depth.append(a)
                np.savez("depth_0%d%d%d%d.npz" % (o, k, j, i), arr_0=depth)
a = np.load('F:/Year4 S1/ELEC5306/project2/final/depth_00000.npz')['arr_0'][2]
print(a.shape)



