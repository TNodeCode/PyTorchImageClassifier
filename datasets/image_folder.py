import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


class ImageFolderDataset():
    def __init__(
        self,
        root_dir: str,
        dataset_name: str,
        num_classes: int = None,
        train_dir: str = "train",
        val_dir: str = "val",
        test_dir: str = "test",
        batch_size_train = 16,
        batch_size_val = 4,
        batch_size_test = 4,
        transform_train = None,
        transform_val = None,
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
        if num_classes is None:
            self.num_classes = len(self.get_categories())
        
    def get_train_dir(self):
        return f"{self.root_dir}/{self.dataset_name}/{self.train_dir}"
        
    def get_val_dir(self):
        return f"{self.root_dir}/{self.dataset_name}/{self.val_dir}"
    
    def get_test_dir(self):
        return f"{self.root_dir}/{self.dataset_name}/{self.test_dir}"
    
    def get_categories(self):
        return list(map(lambda x : {"name": x, "dir": x}, os.listdir(self.get_train_dir())))
    
    def get_dataset_train(self):
        return ImageFolder(root=self.get_train_dir(), transform=self.transform_train)
        
    def get_dataloader_train(self):
        return DataLoader(self.get_dataset_train(), batch_size=self.batch_size_train)
        
    def get_dataset_val(self):
        return ImageFolder(root=self.get_val_dir(), transform=self.transform_val)
        
    def get_dataloader_val(self):
        return DataLoader(self.get_dataset_val(), batch_size=self.batch_size_val)
        
    def get_dataset_test(self):
        return ImageFolder(root=self.get_test_dir(), transform=self.transform_val)
        
    def get_dataloader_test(self):
        return DataLoader(self.get_dataset_test(), batch_size=self.batch_size_test)
    
    def summarize(self):
        categories = self.get_categories()
        n_categories = len(categories)

        train_dirs = []
        val_dirs = []
        test_dirs = []

        for i, category in enumerate(categories):
            train_dirs.append(os.path.join(self.get_train_dir(), category["dir"]))
            val_dirs.append(os.path.join(self.get_val_dir(), category["dir"]))
            test_dirs.append(os.path.join(self.get_test_dir(), category["dir"]))
            
        print('The dataset contains:')
        for i, category in enumerate(categories):
            print('\u2022 {}: {:,} training images, {:,} validation images and {:,} test images'.format(category["name"], len(os.listdir(train_dirs[i])), len(os.listdir(val_dirs[i])), len(os.listdir(test_dirs[i]))))
        