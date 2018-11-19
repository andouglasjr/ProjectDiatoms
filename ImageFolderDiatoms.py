from torchvision import datasets
class ImageFolderDiatoms(datasets   .ImageFolder):
    
    def __getitem__(self, index):
        return super(ImageFolderDiatoms, self).__getitem__(index), self.imgs[index]   