import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision import transforms

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

from src.base.torchvision_dataset import TorchvisionDataset
from src.datasets.preprocessing import get_target_label_idx


class HitsDataset(TorchvisionDataset):
  def __init__(self, root: str, normal_class=1):
    super().__init__(root)

    self.n_classes = 2  # 0: normal, 1: outlier
    self.normal_classes = tuple([normal_class])
    self.outlier_classes = list(range(0, 2))
    self.outlier_classes.remove(normal_class)

    self.data_dict = pd.read_pickle(self.root)
    # hardcoded selected channel
    images = self.normalize_by_image(self.data_dict['images'])[..., 3][
      ..., np.newaxis]
    labels = np.array(self.data_dict['labels'])

    train_array, test_array, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.3, random_state=42)

    transform = transforms.Compose([transforms.ToTensor()])

    target_transform = transforms.Lambda(
        lambda x: int(x in self.outlier_classes))

    train_set = Hits(train_array, train_labels, transform=transform,
                     target_transform=target_transform)
    train_idx_normal = get_target_label_idx(
        np.array(train_set.label_arr), self.normal_classes)
    self.train_set = Subset(train_set, train_idx_normal)
    print(self.train_set.__len__())
    self.test_set = Hits(test_array, test_labels, transform=transform,
                         target_transform=target_transform)

  def normalize_by_image(self, images):
    images -= np.nanmin(images, axis=(1, 2))[:, np.newaxis, np.newaxis, :]
    images = images / np.nanmax(images, axis=(1, 2))[
                      :, np.newaxis, np.newaxis, :]
    return images


class Hits(Dataset):
  def __init__(self, images, labels, transform, target_transform):
    """
    """
    # Transforms
    self.transform = transform
    self.target_transform = target_transform

    self.image_arr = images
    self.label_arr = labels
    print(self.image_arr.shape)
    self.data_len = self.label_arr.shape[0]

  def __getitem__(self, index):
    single_image = self.image_arr[index]
    single_image_label = self.label_arr[index]

    if self.transform is not None:
      img = self.transform(single_image)

    if self.target_transform is not None:
      target = self.target_transform(single_image_label)

    return img, target, index  # only line changed

  def __len__(self):
    return self.data_len
