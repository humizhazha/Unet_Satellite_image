import matplotlib.pyplot as plt
import os
## change your data path here

def drawLossGraph(data_path):
    loss_CE = []
    loss_FK = []
    loss_FM = []
    loss_UL = []
    loss_GAN = []
    f = open(os.path.join(data_path + 'Avg_Train_loss_CE.txt'), "r")
    for x in f:
        loss_CE.append(float(x))
    plt.plot(loss_CE, label="Train_loss_CE")
    f.close()
    f = open(os.path.join(data_path + 'Avg_Train_loss_FK.txt'), "r")
    for x in f:
        loss_FK.append(float(x))
    plt.plot(loss_FK, label="Train_loss_FK")
    f.close()
    f = open(os.path.join(data_path + 'Avg_Train_loss_FM.txt'), "r")
    for x in f:
        loss_FM.append(float(x))
    plt.plot(loss_FM, label="Train_loss_FM")
    f.close()
    f = open(os.path.join(data_path + 'Avg_Train_loss_UL.txt'), "r")
    for x in f:
        loss_UL.append(float(x))
    plt.plot(loss_UL, label="Train_loss_UL")
    f.close()
    linestyle = ':'
    f = open(os.path.join(data_path + 'Avg_Val_loss_GAN.txt'), "r")
    for x in f:
        loss_GAN.append(float(x))
    plt.plot(loss_GAN, label="Validation_loss_GAN", linestyle=linestyle)
    plt.xlabel('Epoch')
    plt.ylabel('Values')
    f.close()
    plt.legend()
    plt.savefig(data_path + 'loss.png')

