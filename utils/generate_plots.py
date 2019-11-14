import os

import matplotlib.pyplot as plt

from pylab import rcParams
rcParams['figure.figsize'] = 12, 8


def draw_Unet_GAN_loss_curves(data_path, sampling_rate):
    '''
     Plot curves for the Unet_GAN and export the figures to the given directory
    :param data_path: the path of the directory that contains the following files:
           (1) 'Avg_Train_loss_CE.txt': average training losses for labeled images
           (2) 'Avg_Train_loss_FK.txt': averages training losses for fake images
           (3) 'Avg_Train_loss_UL.txt': average training losses for unlabeled images
           (4) 'Avg_Val_loss_GAN.txt': average validation losses
    :return: void
    '''

    # read different losses
    with open(os.path.join(data_path, 'Avg_Train_loss_CE.txt'), 'r') as f:
        losses = [float(x.strip()) for x in f]
        plt.plot(losses, 'r', linewidth=2, label="Discriminator loss (labeled images)")

    with open(os.path.join(data_path, 'Avg_Train_loss_FK.txt'), 'r') as f:
        losses = [float(x.strip()) for x in f]
        plt.plot(losses, 'm', linewidth=2, label="Discriminator loss (fake images)")

    with open(os.path.join(data_path, 'Avg_Train_loss_UL.txt'), "r") as f:
        losses = [float(x.strip()) for x in f]
        plt.plot(losses, 'c', linewidth=2, label="Discriminator loss (unlabled images)")

    with open(os.path.join(data_path, 'Avg_Train_loss_FM.txt'), "r") as f:
        losses = [float(x.strip()) for x in f]
        plt.plot(losses, 'k', label="Generator loss (generated images)", )

    with open(os.path.join(data_path, 'Avg_Val_loss_GAN.txt'), "r") as f:
        losses = [float(x.strip()) for x in f]
        plt.plot(losses, 'b', label="Validation loss", linestyle='-.')

    plt.title('Losses for Unet-GAN (1 unlabled image)', fontsize=18)
    plt.xlabel('Epochs (sampled at every {} epoches)'.format(sampling_rate), fontsize=16)
    plt.ylabel('Loss values', fontsize=16)
    plt.legend()
    plt.savefig(os.path.join(data_path, 'unet_gan_loss.png'))
    print('Figure exported to ' + os.path.join(data_path, 'unet_gan_loss.png'))


def draw_Unet_loss_curves(data_path, sampling_rate):
    '''
     Plot curves for the Unet_GAN and export the figures to the given directory
        :param data_path: the path of the directory that contains the following files:
               (1) 'Avg_Train_loss.txt': average training losses
               (2) 'Avg_Validation_loss.txt': validation losses
        :return: void
    '''
    with open(os.path.join(data_path, 'Avg_Train_loss.txt'), "r") as f:
        losses = [float(x.strip()) for x in f]
        plt.plot(losses, 'r', label="Training loss")

    with open(os.path.join(data_path, 'Avg_Val_loss.txt'), "r") as f:
        losses = [float(x.strip()) for x in f]
        plt.plot(losses, 'b', label="Validation loss", linewidth=2)

    plt.title('Losses for Unet', fontsize=18)
    plt.xlabel('Epochs (sampled at every {} epoches)'.format(sampling_rate), fontsize=16)
    plt.savefig(os.path.join(data_path, 'unet_loss.png'))
    print('Figure exported to ' + os.path.join(data_path, 'unet_loss.png') )


if __name__ == '__main__':

    #draw_Unet_GAN_loss_curves(os.path.join('..', 'Unet_GAN', 'results', 'crops', '1-unlabeled'), sampling_rate=5)
    draw_Unet_loss_curves(os.path.join('..', 'Unet', 'results', 'crops', 'Exp_2_baseline'), sampling_rate=5)



