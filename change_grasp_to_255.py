import numpy as np


for o in range(0,6):
    for k in range(0,10):
        for j in range(0,10):
            for i in range(0,10):
                first_grasp = np.load('E:/Year4 S2/毕设/grasp/depth_ims_tf_table_0%d%d%d%d.npz' % (o,k,j,i))['arr_0']
                final_grasp = []
                for a in range(0, 1000):
                    grasp = first_grasp[a, ...]
                    #print(grasp.shape)
                    new_grasp = (grasp.max() - grasp) * (255/(grasp.max() - grasp.min()))
                    new_grasp = new_grasp.astype(int)
                    final_grasp.append(new_grasp)
                np.savez("grasp_255_0%d%d%d%d.npz" % (o,k,j,i), grasp=final_grasp)
