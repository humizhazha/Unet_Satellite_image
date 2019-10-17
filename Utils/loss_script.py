import matplotlib.pyplot as plt
import os
## change your data path here
data_path = '../data/'
## change your target file here
file_name = 'Val_loss.txt'
loss = []

f = open(os.path.join(data_path+file_name), "r")
for x in f:
    loss.append(float(x))
plt.plot(loss, label = "line 1")
plt.savefig(file_name+'.png')