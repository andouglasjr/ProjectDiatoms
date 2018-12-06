from torchvision import datasets
from DatasetFolderDiatoms import DatasetFolderDiatoms

from PIL import Image

import os
import os.path

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

class ImageFolderDiatoms(DatasetFolderDiatoms):
    
    def __init__(self, root, transform=None, loader = default_loader, target_transform = None, number_by_class = None):
        super(ImageFolderDiatoms, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform, number_by_class = number_by_class)
        self.imgs = self.samples
        self.number_by_class = number_by_class
    
    def __getitem__(self, index):
        return super(ImageFolderDiatoms, self).__getitem__(index), self.imgs[index]
