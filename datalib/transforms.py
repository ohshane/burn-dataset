import torchvision.transforms as T
from PIL import ImageOps


def Compose(transforms):
    def _f(x):
        for t in transforms:
            x = t(x)
        return x
    return _f

def exif_transpose():
    return ImageOps.exif_transpose

def squish_resize(size):
    return T.Resize((size,size))

def pad_resize(size, color=(0,0,0)):
    def _f(x):
        x = ImageOps.contain(x, (size,size))
        x = ImageOps.pad(x, (size,size), color=color)
        return x
    return _f

def random_contained_crop(size):
    return T.Compose([
        T.Resize(size),
        T.RandomCrop(size)
    ])

def contained_crop(size):
    return T.Compose([
        T.Resize(size),
        T.CenterCrop(size)
    ])

def to_tensor():
    return T.ToTensor()

def normalize(mode='imagenet'):
    assert mode in ['imagenet', 'skinex']

    mean = [0,0,0]
    std  = [1,1,1]
    if mode == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
    elif mode == 'skinex':
        mean = [0.62069, 0.39049, 0.39372]
        std  = [0.04334, 0.06039, 0.05937]
    
    return T.Normalize(mean, std)

