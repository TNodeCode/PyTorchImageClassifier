import os
import torch
import torchvision
from torchvision.transforms import v2
from datetime import datetime
from models.classifier import Classifier
from tqdm import tqdm
import logger
import inference


class AbstractClassifier():
    def __init__(
        self,
        name: str,
        num_classes: int,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        log_dir: str = None,
        fine_tuning: bool = True,
    ):
        self.name = name
        self.num_classes = num_classes
        self.fine_tuning = fine_tuning
        self.device = device
        self.model = self.build_model(num_classes=num_classes, resume=resume, device=device, fine_tuning=fine_tuning).to(device)
        if resume:
            self.load_weights(resume)
        if root_dir is None:
            self.root_dir = os.getcwd()
        else:
            self.root_dir = root_dir
        self.log_dir = log_dir
        
    def inference(self):
        return inference.Inference(detector=self)
        
    def get_loss_names(self) -> list[str]:
        return ["loss_classification"]
        
    def get_weight_filename(self, epoch: int):
        return f"{self.name}_epoch{str(epoch).zfill(4)}.pth"
    
    def load_pretrained_model(self):
        raise NotImplementedError("this function needs to be implemented")
        
    def load_weights(self, filepath):
        self.model.load_state_dict(
            torch.load(filepath, map_location=torch.device(self.device))["model_state"]
        )
        
    def replace_head(self, model, num_classes: int):
        raise NotImplementedError("this function needs to be implemented")
    
    def build_model(
        self,
        num_classes: int = None,
        resume: str = None,
        fine_tuning: bool = True,
        device: str = "cpu"
    ):
        ### Instantiate model
        model = self.load_pretrained_model()
        
        ### fine-tuning
        for param in model.parameters():
            param.requires_grad = fine_tuning

        ### Replace the head of the network
        if num_classes:
            self.replace_head(model, num_classes)
            
        return model.to(self.device)
            
    def create_log_dir(self):
        if self.log_dir is None:
            log_dir = os.path.join(self.root_dir, "logs", self.name)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            n_models = len(os.listdir(log_dir))
            self.log_dir = os.path.join(log_dir, f"training_{n_models+1}")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)  

    def init_epoch_losses(self) -> dict:
        loss_dict = {"loss": 0.0}
        for k in self.get_loss_names():
            if k[0:5] != "loss_":
                k = "loss_" + k
            loss_dict |= {k: 0.0}
        return loss_dict

    def update_epoch_losses(self, epoch_losses: dict, loss_dict: dict):
        loss_total = (sum(v for v in loss_dict.values()))
        epoch_losses["loss"] += loss_total
        for k in self.get_loss_names():
            loss_k = loss_dict[k]
            if k[0:5] != "loss_":
                k = "loss_" + k
            epoch_losses[k] += loss_k
        return epoch_losses
    
    def log_epoch_metrics(self, n_epochs, epoch, epoch_metrics):
        epoch_metrics |= {"epoch": epoch}
        log_items = []
        for key in epoch_metrics.keys():
            log_items.append(f"{key}={epoch_metrics[key]}")
        log_text = f"Epoch {epoch}/{n_epochs}: "
        log_text += (", ".join(log_items))
        # Save epoch logs on disk
        logger.log_epoch(os.path.join(self.log_dir, "epochs.yaml"), epoch_metrics)
        print(f"Epoch {epoch}/{n_epochs}: {log_text}")
        
    def extract_images_and_targets(self, data):
        raise NotImplementedError("this function needs to be implemented")
        
    def run_training_data_through_model(self, data):
        # Compute loss of batch
        return self.model(data)
        
    def train_one_epoch(self, train_dataloader, optim, criterion) -> dict:
        # Initialize epoch losses
        epoch_losses = self.init_epoch_losses()
        n_correct = 0
        n_images = 0
        self.model.train()
        # Set format for progress bar.
        bar_format = '{l_bar}{bar:20}{r_bar}{bar:-10b}'
        for images, labels in tqdm(train_dataloader, bar_format=bar_format, total=len(train_dataloader), desc="Training batches"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            optim.zero_grad()
            output = self.run_training_data_through_model(images)
            loss = criterion(output, labels)
            loss.backward()
            optim.step()
            # Compute accuracy
            topv, topi = torch.topk(output, k=1)
            n_images += images.shape[0]
            n_correct += (topi.squeeze() == labels).sum().item()
            epoch_losses = self.update_epoch_losses(
                epoch_losses,
                {"loss_classification": loss.item()}
            )
        return {"n_train_images": n_images, "train_accuracy": n_correct/n_images,**epoch_losses}

    def train(
        self,
        n_epochs: int,
        lr: float,
        start_epoch: int = 0,
        resume: str = None,
        save_every: int = None,
        lr_step_every: int = 20,
        num_classes = None,
        device="cpu",
        log_dir: str = None,
        dataset = None,
        criterion = None,
        optim = None,
        lr_scheduler = None,
    ):
        # Set log directory
        if log_dir is not None:
            self.log_dir = log_dir
        
        # check if weights from previous epoch are available
        if resume is None and start_epoch and start_epoch > 0:
            resume = self.root_dir + "/" + self.get_weight_filename(start_epoch)
        
        # Criterion
        if criterion is None:
            criterion = torch.nn.NLLLoss()

        # Optimizer
        if optim is None:
            optim = torch.optim.Adam(self.model.parameters(), lr=lr)
            
        # Learning rate scheduler
        if lr_scheduler is None:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.1)
        
        # Build log directory
        self.create_log_dir()
        print(f"Logging training at {self.log_dir}")

        # Move model to training device
        self.model.to(self.device)
        print("Start training ...")
        
        # Create dataloaders
        dataloader_train = dataset.get_dataloader_train()
        dataloader_val = dataset.get_dataloader_val()

        for epoch in range(start_epoch+1, n_epochs+1):
            # Put model into training mode
            self.model.train()
            
            # Train the model for one epoch
            print("Train ...")
            epoch_metrics = {"learning_rate": lr_scheduler.get_last_lr()[0],
                             "lr_step_every": lr_step_every,
                             "optim": str(type(optim)),
                             "scheduler": str(type(lr_scheduler)),
                             "epoch_start": datetime.now().astimezone().isoformat(),
                             "batch_size": dataset.batch_size_train}
            epoch_metrics |= self.train_one_epoch(dataloader_train, optim, criterion)          
            epoch_metrics |= {"epoch_end": datetime.now().astimezone().isoformat()}
            
            # Save model weights
            if save_every and epoch > 0 and epoch % save_every == 0:
                torch.save(
                    {"name": self.name,
                     "num_classes": self.num_classes,
                     "model_state": self.model.state_dict(),
                     "optim_state": optim.state_dict(),
                     "scheduler_state": lr_scheduler.state_dict(),
                     **epoch_metrics,
                    },
                    os.path.join(self.log_dir, self.get_weight_filename(epoch))
                )
                
            # Update learning rate
            if epoch > 0 and epoch % lr_step_every == 0:
                lr_scheduler.step()
                
            # Validate model
            if dataloader_val is not None:
                print("Validating ...")
                # Put model into evaluation mode
                self.model.eval()
                n_images = 0
                n_correct = 0
                bar_format = '{l_bar}{bar:20}{r_bar}{bar:-10b}'
                for images, labels in tqdm(dataloader_val, bar_format=bar_format, total=len(dataloader_val), desc="Validation batches"):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    # Run images through model
                    output = self.run_training_data_through_model(images)
                    # Compute accuracy
                    topv, topi = torch.topk(output, k=1)
                    n_images += images.shape[0]
                    n_correct += (topi.squeeze() == labels).sum().item()
                    epoch_metrics |= {"n_val_images": n_images, "val_accuracy": n_correct/n_images}
            self.log_epoch_metrics(n_epochs=n_epochs, epoch=epoch, epoch_metrics=epoch_metrics)
            
            
class MobileNetV3SmallClassifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="mobilenet_v3_small",
            num_classes=num_classes,
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=True,
        )

    def load_pretrained_model(self):
        return torchvision.models.mobilenet_v3_small(weights='DEFAULT')
    
    def replace_head(self, model, num_classes: int):
        model.classifier= Classifier(
            input_size=576,
            output_size=num_classes,
            hidden_sizes=[],
            softmax_dim=1,
        )
            
            
class ResNet50Classifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="resnet50",
            num_classes=num_classes,
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=True,
        )

    def load_pretrained_model(self):
        return torchvision.models.resnet50(weights='DEFAULT')
    
    def replace_head(self, model, num_classes: int):
        model.fc= Classifier(
            input_size=2048,
            output_size=num_classes,
            hidden_sizes=[],
            softmax_dim=1,
        )
            
            
class SwinTClassifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="swin_t",
            num_classes=num_classes,
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=True,
        )

    def load_pretrained_model(self):
        return torchvision.models.swin_t(weights='DEFAULT')
    
    def replace_head(self, model, num_classes: int):
        model.head= Classifier(
            input_size=768,
            output_size=num_classes,
            hidden_sizes=[],
            softmax_dim=1,
        )
