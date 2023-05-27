import matplotlib.pyplot as plt
import pathlib
import imagesize
from PIL import Image
import numpy as np

def plot_example(leaf_type, directory):
    '''
    Plots first 4 images of corn leaf
    ---Parameters---
    leaf_type (str) Corn disease type to title image
    directory (pathlib.WindowsPath object)
    ---Returns---
    None
    '''
    plt.suptitle(leaf_type+' Example')
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.imshow(plt.imread(list(directory.glob(leaf_type+'/*.jpg'))[i]))
    return None

def min_max_dimensions(directory):
    width_max = 0
    height_max = 0
    width_min=np.inf
    height_min=np.inf
    for image_path in directory.glob('*/*.jpg'):
        width, height = imagesize.get(image_path)
        if width > width_max:
            width_max = width
        if height > height_max:
            height_max = height
        if width < width_min:
            width_min = width
        if height < height_min:
            height_min = height
    return ((width_max, height_max), (width_min, height_min))

def reshape(PIL_img, new_width, new_height, color):
    width, height = PIL_img.size
    result = Image.new(PIL_img.mode, (new_width, new_height), color)
    left=(new_width-width)//2
    top=(new_height-height)//2
    result.paste(PIL_img, (left, top))
    return result
    
def dimensions_distributions(directory):
    widths = []
    heights = []
    for image_path in directory.glob('*/*.jpg'):
        width, height = imagesize.get(image_path)
        widths.append(width)
        heights.append(height)
    return (widths, heights)    
    
    
    
    
  