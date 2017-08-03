from scipy.misc import imread, imsave
import torch

def load_image(image_filepath):
    image = imread(image_filepath)
    image = torch.from_numpy(image)
    return image

def save_image(filename, image):
    imsave(filename, image)

def is_image(self, image_filename):
    return any(image_filename.endswith(extension) for extension in [".png"])
