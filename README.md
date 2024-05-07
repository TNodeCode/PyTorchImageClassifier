# PyTorch Image Classifier

This repository provides you an easy to use API for training PyTorch image classification models.

## How to train a model

First you need to import the necessary libraries:

```python
import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder

from datasets.image_folder import ImageFolderDataset
from classifiers import *
```

Next create augmentation pipelines for the training and the validation datasets. Also you need to create an instance of `ImageFolderDataset`.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

transform_train = v2.Compose([
    v2.RandomResizedCrop(224, scale=(0.7, 1.0)),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomRotation(45),
    v2.ToTensor(),
    v2.ToDtype(torch.float32),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_val = v2.Compose([
    v2.RandomResizedCrop(224, scale=(1.0, 1.0)),
    v2.ToTensor(),
    v2.ToDtype(torch.float32),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = ImageFolderDataset(
    root_dir = os.getcwd() + "/images",
    dataset_name = "leafes",
    batch_size_train = 16,
    batch_size_val = 16,
    transform_train = transform_train,
    transform_val = transform_val
)
```

Instantiate one of the classifiers that can be found in `classifiers.py`. Here is an example for the MobileNetV3 classifier:

```python
classifier = MobileNetV3SmallClassifier(
    num_classes=dataset.num_classes,
    device=device,
    fine_tuning=True
)
```

The final step is to call the `train` method of the classifier.

```python
classifier.train(
    n_epochs = 50,
    lr = 1e-3,
    start_epoch = 0,
    resume = None,
    save_every = 50,
    lr_step_every = 10,
    dataset=dataset,
    num_classes = dataset.num_classes,
    device=device,
    log_dir=os.path.join(os.getcwd(), "logs", dataset.dataset_name, classifier.name),
)
```
