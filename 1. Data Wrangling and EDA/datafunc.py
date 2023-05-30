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
    '''
    Finds the maximum and minimum dimensions of images within a directory
    ---Parameters---
    directory (pathlib.WindowsPath object)
    ---Returns---
    Maximum and minimum image dimensions in the form ((width_max, height_max), (width_min, height_min))
    '''
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
    '''
    Reshapes an image to new specified dimenions without changing aspect ratio by padding the sides
    ---Parameters---
    PIL_img (PIL.Image class) Image to reshape
    new_width (int) new width of image after reshaping, must be greater than or equal to current image width
    new_height (int) new height of image after reshaping, must be greater than or equal to current image height
    color (3-tuple) color of pixels used to pad the image in (R, G, B) format
    ---Returns---
    result (PIL.Image class) new image after reshaping with padding
    '''
    width, height = PIL_img.size
    result = Image.new(PIL_img.mode, (new_width, new_height), color)
    left=(new_width-width)//2
    top=(new_height-height)//2
    result.paste(PIL_img, (left, top))
    return result
    
def dimensions_distributions(directory):
    '''
    Gets the distribution of the pixel widths and pixel heights of all images in a specified directory
    ---Parameters---
    directory (pathlib.WindowsPath object) directory of images
    ---Returns---
    2-tuple where each element is a list in the form (widths, heights)
    '''
    widths = []
    heights = []
    for image_path in directory.glob('*/*.jpg'):
        width, height = imagesize.get(image_path)
        widths.append(width)
        heights.append(height)
    return (widths, heights)

def plot_dim_dist(data, bins=10, dim ='Widths', range_ = False):
    '''
    Plots histogram of width/height 
    ---Parameters---
    data (array) data to plot 
    bins (int) number of bins the histogram should have
    dim (str) what dimension width/height is being plotted, for titling purposes
    range_ (2-tuple) range of x axis values for the plot to take in the form (min, max)
    ---Returns---
    None
    '''
    if range_ == False:
        plt.hist(data, bins = bins)
    else:
        plt.hist(data, bins = bins, range= range_)
    plt.title('Distribution of Image '+ dim)
    plt.xlabel('Image' + dim + '(Pixels)')
    plt.ylabel('Frequency (absolute)');
    return None
    
def avg_image(class_, directory):
    '''
    Plots color averaged image of specific corn leaf class
    ---Parameters---
    class_ (str) name of directory of stored class images
    directory (pathlib.WindowsPath object) parent directory where images are stored
    ---Returns---
    None
    '''
    ims=[]
    for img in directory.glob(class_ + '/*.jpg'):
        ims.append(Image.open(img))
    ims = np.array([np.array(im) for im in ims])
    ims_avg = np.average(ims, axis=0)
    result = Image.fromarray(ims_avg.astype('uint8'))
    plt.imshow(result)
    plt.title('Average ' + class_ + ' Image')
    plt.show()
    return None
    
    
    
  