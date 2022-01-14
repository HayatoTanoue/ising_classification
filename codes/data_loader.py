import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from glob import glob


def set_transform(resize):
    """画像前処理設定"""
    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ]
    )
    return transform


# for cnn loader
def cnn_data_loader_cv(name, resize, path=None):
    """合成データセット"""
    if path is None:
        path = f"../data/{name}/binary"
    num_class = len(glob(f"{path}/*"))
    transforms = set_transform(resize)
    dataset = ImageFolder(path, transforms)

    return dataset, num_class
