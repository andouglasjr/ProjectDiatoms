import csv
import numpy as np
from matplotlib import pyplot as plt

def plot_log(filename, show=True):
    # load data
    keys = []
    values = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if keys == []:
                for key, value in row.items():
                    keys.append(key)
                    values.append(float(value))
                continue

            for _, value in row.items():
                values.append(float(value))

        values = np.reshape(values, newshape=(-1, len(keys)))

    fig = plt.figure()
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(121)

    epoch_axis = 0
    for i, key in enumerate(keys):
        if key == 'epoch':
            epoch_axis = i
            values[:, epoch_axis] += 1
            break
    for i, key in enumerate(keys):
        if key.find('loss') >= 0:  # loss
            print(values[:, i])
            if (key == 'loss'):
            	label = 'Training Loss' 
            else: 
            	label = 'Validation Loss'
            plt.plot(values[:, epoch_axis], values[:, i], label=label, linewidth=2.0)
    plt.legend()
    plt.grid()
    plt.title('Loss Function')

    fig.add_subplot(122)
    for i, key in enumerate(keys):
        if key.find('acc') >= 0:  # acc
        	if (key == 'train_acc'):
        		label = 'Training Acc'
        	else:
        		label = 'Validation Acc'
        	plt.plot(values[:, epoch_axis], values[:, i], label=label, linewidth=2.0)
    plt.legend()
    plt.grid()
    plt.title('Accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show()


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image


if __name__=="__main__":
    plot_log('log_resnet_101_50_classes.csv')

