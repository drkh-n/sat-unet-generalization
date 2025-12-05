# import imgaug.augmenters as iaa
from torchvision.transforms import v2

STD_TRANSFROMS = v2.Compose([
    # v2.RandomResizedCrop(size=(384, 384), scale=(0.9, 1.1)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),   
    v2.RandomRotation(degrees=15),
])