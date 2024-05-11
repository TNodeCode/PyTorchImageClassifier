import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder

from datasets.multi_class_images import MultiClassImageDataset
from classifiers import *

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

dataset = MultiClassImageDataset(
    root_dir = os.getcwd() + "/images",
    dataset_name = os.getenv("DATASET_NAME"),
    batch_size_train = int(os.getenv("BATCH_SIZE")),
    batch_size_val = int(os.getenv("BATCH_SIZE_VAL")),
    transform_train = transform_train,
    transform_val = transform_val
)

classifier = MobileNetV3LargeClassifier(
    num_classes=dataset.num_classes,
    device=device,
    fine_tuning=True,
    multi_class=True,
)

criterion = torch.nn.BCEWithLogitsLoss()

classifier.train(
    n_epochs = int(os.getenv("EPOCHS")),
    lr = 1e-3,
    start_epoch = 0,
    resume = None,
    save_every = 5,
    lr_step_every = 10,
    threshold_true_prediction=0.9,
    weight_true_labels=0.9,
    weight_false_labels=0.1,
    dataset=dataset,
    criterion=criterion,
    num_classes = dataset.num_classes,
    device=device,
    log_dir=os.path.join(os.getcwd(), "logs", dataset.dataset_name, classifier.name),
)