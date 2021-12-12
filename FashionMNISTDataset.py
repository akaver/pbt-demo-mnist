from torchvision.datasets import FashionMNIST
from typing import Optional, Callable, Tuple, Any
import torchvision.transforms.functional as TF
from PIL import Image


# https://github.com/zalandoresearch/fashion-mnist
# https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py
class FashionMNISTDataset(FashionMNIST):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            conf={}
    ) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.conf = conf

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        # do not apply outside transformation to image
        """
        if self.transform is not None:
            img = self.transform(img)
        """

        # converts byte values to 0.0-1.0 float tensor
        img = TF.to_tensor(img)

        if self.conf is not None and "data_mean" in self.conf and "data_std" in self.conf:
            img = TF.normalize(img, self.conf["data_mean"], self.conf["data_std"])

        # pixel value to use when transformation needs to fill empty space (on rotation for example)
        # use corner pixel, should be black
        fill_pixel = img[0, 0, 0].item()

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def main():
    pass


if __name__ == '__main__':
    main()
