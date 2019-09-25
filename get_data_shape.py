import h5py
import os
import numpy as np
f = h5py.File(os.path.join("../data", 'train_label.h5'), 'r')
label_vols = np.array(f['train'])

label = np.array(f['train_mask'])[:, 6]

print(label_vols.shape)
print(label.shape)
# s = ((2,2))
# s = np.zeros(s)
# s+=1
# print(np.min(s))