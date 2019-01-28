from torchvision import datasets
import torch.utils.data as data

from PIL import Image

import os
import os.path  

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions, number_by_class):
    images = []
    dir = os.path.expanduser(dir)
    cont_number = 1
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
                
                    if number_by_class > 0:
                        if cont_number == number_by_class:
                            cont_number = 1
                            break
                        cont_number += 1      
                    
    return images

class DatasetFolderDiatoms(datasets.DatasetFolder):
    
    def __init__(self, root, loader, extensions, transform=None, target_transform=None, number_by_class = None):
        super(DatasetFolderDiatoms, self).__init__(root, loader, extensions, transform=None, target_transform=None)
        classes, class_to_idx = find_classes(root)
        #print(number_by_class)
        samples = make_dataset(root, class_to_idx, extensions, number_by_class)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform