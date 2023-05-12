import matplotlib.pyplot as plt
import pathlib

def plot_example(leaf_type, directory):
    plt.suptitle(leaf_type+' Example')
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.imshow(plt.imread(list(directory.glob(leaf_type+'/*.jpg'))[i]))
    return None