import os
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class MultiClassImageDataset(Dataset):
    def __init__(
            self,
            root_dir,
            dataset_name,
            train_dir: str = "train",
            val_dir: str = "val",
            test_dir: str = "test",
            batch_size_train = 16,
            batch_size_val = 4,
            batch_size_test = 4,
            transform_train=None,
            transform_val=None
            ):
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test
        self.transform_train = transform_train
        self.transform_val = transform_val
        df_data = pd.read_csv(f"{self.root_dir}/{dataset_name}/dataset.csv")
        self.filenames = df_data.loc[:, "image"]
        self.labels = df_data.drop("image", axis=1).to_numpy().astype(np.float32)
        self.num_classes = self.labels.shape[1]

    def __len__(self):
        return self.filenames.shape[0]

    def __getitem__(self, idx):
        image_path = self.filenames[idx]
        labels = self.labels[idx]
        image = Image.open(f"{self.root_dir}/{self.dataset_name}/images/{image_path}").convert("RGB")

        if self.transform_train is not None:
            image = self.transform_train(image)

        return image, labels
    
        
    def get_dataloader_train(self):
        return DataLoader(self.get_dataset_train(), batch_size=self.batch_size_train)
        
    def get_train_dir(self):
        return f"{self.root_dir}/{self.dataset_name}/images/{self.train_dir}"
        
    def get_val_dir(self):
        return f"{self.root_dir}/{self.dataset_name}/images/{self.val_dir}"
    
    def get_test_dir(self):
        return f"{self.root_dir}/{self.dataset_name}/images/{self.test_dir}"
    
    def get_categories(self):
        return list(map(lambda x : {"name": x, "dir": x}, os.listdir(self.get_train_dir())))
    
    def get_dataset_train(self):
        return MultiClassImageDataset(
            root_dir=self.root_dir,
            dataset_name=self.dataset_name,
            train_dir=self.train_dir,
            val_dir=self.val_dir,
            test_dir=self.test_dir,
            batch_size_train=self.batch_size_train,
            batch_size_val=self.batch_size_val,
            batch_size_test=self.batch_size_test,
            transform_train=self.transform_train,
            transform_val=self.transform_val,
        )
        
    def get_dataloader_train(self):
        return DataLoader(self.get_dataset_train(), batch_size=self.batch_size_train)
        
    def get_dataset_val(self):
        return MultiClassImageDataset(
            root_dir=self.root_dir,
            dataset_name=self.dataset_name,
            train_dir=self.train_dir,
            val_dir=self.val_dir,
            test_dir=self.test_dir,
            batch_size_train=self.batch_size_train,
            batch_size_val=self.batch_size_val,
            batch_size_test=self.batch_size_test,
            transform_train=self.transform_train,
            transform_val=self.transform_val,
        )
        
    def get_dataloader_val(self):
        return DataLoader(self.get_dataset_val(), batch_size=self.batch_size_val)
        
    def get_dataset_test(self):
        return MultiClassImageDataset(
            root_dir=self.root_dir,
            dataset_name=self.dataset_name,
            train_dir=self.train_dir,
            val_dir=self.val_dir,
            test_dir=self.test_dir,
            batch_size_train=self.batch_size_train,
            batch_size_val=self.batch_size_val,
            batch_size_test=self.batch_size_test,
            transform_train=self.transform_train,
            transform_val=self.transform_val,
        )
        
    def get_dataloader_test(self):
        return DataLoader(self.get_dataset_test(), batch_size=self.batch_size_test)
