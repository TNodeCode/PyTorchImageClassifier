import os
import torch
import torchvision
from torchvision.transforms import v2
from datetime import datetime
from models.classifier import Classifier, LogitsHead, ViTLogitsHead
from tqdm import tqdm
import logger
import inference
import models.vit


class AbstractClassifier():
    def __init__(
        self,
        name: str,
        num_classes: int,
        head_input_dim: int = None,
        head_attribute_name: str = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        log_dir: str = None,
        fine_tuning: bool = True,
    ):
        self.name = name
        self.num_classes = num_classes
        self.multi_class = multi_class
        self.head_input_dim = head_input_dim
        self.head_attribute_name = head_attribute_name
        self.fine_tuning = fine_tuning
        self.device = device
        self.model = self.build_model(num_classes=num_classes, resume=resume, fine_tuning=fine_tuning).to(device)
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
        if self.head_input_dim is None:
            raise ValueError("You must specify an input dimension for the model's head")
        if self.head_attribute_name is None:
            raise ValueError("You must specify the attribute name where the head is stored in")
        if self.multi_class:
            setattr(model, self.head_attribute_name, LogitsHead(
                input_size=self.head_input_dim,
                output_size=num_classes,
                hidden_sizes=[],
            ))
        else:
            setattr(model, self.head_attribute_name, Classifier(
                input_size=self.head_input_dim,
                output_size=num_classes,
                hidden_sizes=[],
                softmax_dim=1,
            ))
    
    def build_model(
        self,
        num_classes: int = None,
        resume: str = None,
        fine_tuning: bool = True,
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
        
    def train_one_epoch(self, train_dataloader, optim, criterion, weight_true_labels, weight_false_labels) -> dict:
        epoch_metrics = {"train_true_positives": 0, "train_false_positives": 0, "train_true_negatives": 0, "train_false_negatives": 0}
        # Initialize epoch losses
        epoch_losses = self.init_epoch_losses()
        true_positives = 0
        n_images = 0
        self.model.train()
        # Set format for progress bar.
        bar_format = '{l_bar}{bar:20}{r_bar}{bar:-10b}'
        for images, labels in tqdm(train_dataloader, bar_format=bar_format, total=len(train_dataloader), desc="Training batches"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            optim.zero_grad()
            output = self.run_training_data_through_model(images)
            if self.multi_class:
                # creatte masks for true and false labels
                mask_true_labels = labels == 1.0
                mask_false_labels = labels == 0.0
                # compute loss for positive and negative samples
                loss_true_labels = criterion(output[mask_true_labels], labels[mask_true_labels])
                loss_false_labels = criterion(output[mask_false_labels], labels[mask_false_labels])
                # compute total loss based on weighted losses for positive and negative samples
                loss = weight_true_labels * loss_true_labels + weight_false_labels * loss_false_labels
            else:
                loss = criterion(output, labels)
            loss.backward()
            optim.step()
            # Compute accuracy
            topv, topi = torch.topk(output, k=1)
            n_images += images.shape[0]
            if self.multi_class:
                # threshold for accepting a prediction as true
                threshold = 0.5
                # Run logits of neuran network through sigmoid layer and checking if values are greater than the threshold
                predictions = torch.sigmoid(output.detach()) >= threshold
                # create masks for true and false predictions
                mask_true_predictions = (predictions == 1.0)
                mask_false_predictions = (predictions == 0.0)
                # compute TP, TN, FP and FN
                true_positives = (predictions[mask_true_predictions] == labels[mask_true_predictions]).sum().item()
                false_positives = (predictions[mask_true_predictions] != labels[mask_true_predictions]).sum().item()
                true_negatives = (predictions[mask_false_predictions] == labels[mask_false_predictions]).sum().item()
                false_negatives = (predictions[mask_false_predictions] != labels[mask_false_predictions]).sum().item()
                # update epoch metrics
                epoch_metrics["train_true_positives"] += true_positives
                epoch_metrics["train_false_positives"] += false_positives
                epoch_metrics["train_true_negatives"] += true_negatives
                epoch_metrics["train_false_negatives"] += false_negatives
            else:
                true_positives += (topi.squeeze() == labels).sum().item()
            epoch_losses = self.update_epoch_losses(
                epoch_losses,
                {"loss_classification": loss.item()}
            )

        # compute accuracy, precision, recall and F1 score
        accuracy = (epoch_metrics["train_true_positives"] + epoch_metrics["train_true_negatives"]) / (epoch_metrics["train_true_positives"] + epoch_metrics["train_true_negatives"] + epoch_metrics["train_false_positives"] + epoch_metrics["train_false_negatives"])
        precision = epoch_metrics["train_true_positives"] / (epoch_metrics["train_true_positives"] + epoch_metrics["train_false_positives"]) if epoch_metrics["train_true_positives"] + epoch_metrics["train_false_positives"] > 0 else 0.0
        recall = epoch_metrics["train_true_positives"] / (epoch_metrics["train_true_positives"] + epoch_metrics["train_false_negatives"])
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
        epoch_metrics["train_accuracy"] = accuracy
        epoch_metrics["train_precision"] = precision
        epoch_metrics["train_recall"] = recall
        epoch_metrics["train_f1_score"] = f1_score

        return {
            "n_train_images": n_images,
            "train_accuracy": true_positives/n_images,
            **epoch_metrics,
            **epoch_losses
        }

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
        weight_true_labels: float = 0.5,
        weight_false_labels: float = 0.5,
        threshold_true_prediction: float = 0.5,
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
            epoch_metrics |= self.train_one_epoch(
                train_dataloader=dataloader_train,
                optim=optim,
                criterion=criterion,
                weight_true_labels=weight_true_labels,
                weight_false_labels=weight_false_labels,
            )          
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
                true_positives = 0
                bar_format = '{l_bar}{bar:20}{r_bar}{bar:-10b}'
                for images, labels in tqdm(dataloader_val, bar_format=bar_format, total=len(dataloader_val), desc="Validation batches"):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    # Run images through model
                    output = self.run_training_data_through_model(images)
                    if self.multi_class:
                        # threshold for accepting a prediction as true
                        threshold = threshold_true_prediction
                        # Run logits of neuran network through sigmoid layer and checking if values are greater than the threshold
                        predictions = torch.sigmoid(output.detach()) >= threshold
                        # create masks for true and false predictions
                        mask_true_predictions = (predictions == 1.0)
                        mask_false_predictions = (predictions == 0.0)
                        # compute TP, TN, FP and FN
                        true_positives = (predictions[mask_true_predictions] == labels[mask_true_predictions]).sum().item()
                        false_positives = (predictions[mask_true_predictions] != labels[mask_true_predictions]).sum().item()
                        true_negatives = (predictions[mask_false_predictions] == labels[mask_false_predictions]).sum().item()
                        false_negatives = (predictions[mask_false_predictions] != labels[mask_false_predictions]).sum().item()
                        # compute accuracy, precision, recall and F1 score
                        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
                        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0.0
                        recall = true_positives / (true_positives + false_negatives)
                        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
                        epoch_metrics |= {"n_val_images": n_images, "val_accuracy": accuracy, "val_precision": precision, "val_recall": recall, "val_f1_score": f1_score}
                    else:
                        # Compute accuracy
                        topv, topi = torch.topk(output, k=1)
                        n_images += images.shape[0]
                        true_positives += (topi.squeeze() == labels).sum().item()
                        epoch_metrics |= {"n_val_images": n_images, "val_accuracy": true_positives/n_images}

            self.log_epoch_metrics(n_epochs=n_epochs, epoch=epoch, epoch_metrics=epoch_metrics)
            
            
class EfficientNetV2SClassifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="efficientnet_v2_s",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=1280,
            head_attribute_name="classifier",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.efficientnet_v2_s(weights='DEFAULT')
            
            
class EfficientNetV2MClassifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="efficientnet_v2_m",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=1280,
            head_attribute_name="classifier",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.efficientnet_v2_m(weights='DEFAULT')
            
            
class EfficientNetV2LClassifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="efficientnet_v2_l",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=1280,
            head_attribute_name="classifier",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.efficientnet_v2_l(weights='DEFAULT')
             
            
class DenseNet121Classifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="densenet121",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=1024,
            head_attribute_name="classifier",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.densenet121(weights='DEFAULT')
             
            
class DenseNet161Classifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="densenet161",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=2208,
            head_attribute_name="classifier",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.densenet161(weights='DEFAULT')
             
            
class DenseNet169Classifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="densenet169",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=1664,
            head_attribute_name="classifier",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.densenet169(weights='DEFAULT')
             
            
class DenseNet201Classifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="densenet201",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=1920,
            head_attribute_name="classifier",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.densenet201(weights='DEFAULT')
             
            
class MobileNetV3SmallClassifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="mobilenet_v3_small",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=576,
            head_attribute_name="classifier",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.mobilenet_v3_small(weights='DEFAULT')
            
            
class MobileNetV3LargeClassifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="mobilenet_v3_large",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=960,
            head_attribute_name="classifier",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.mobilenet_v3_large(weights='DEFAULT')
            
            
class ConvNeXtTinyClassifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="convnext_tiny",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=768,
            head_attribute_name="classifier",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.convnext_tiny(weights='DEFAULT')
    
    def replace_head(self, model, num_classes: int):
        if self.head_input_dim is None:
            raise ValueError("You must specify an input dimension for the model's head")
        if self.head_attribute_name is None:
            raise ValueError("You must specify the attribute name where the head is stored in")
        if self.multi_class:
            model.classifier[2] = LogitsHead(
                input_size=self.head_input_dim,
                output_size=num_classes,
                hidden_sizes=[],
            )
        else:
            model.classifier[2] = Classifier(
                input_size=self.head_input_dim,
                output_size=num_classes,
                hidden_sizes=[],
                softmax_dim=1,
            )
            
            
class ConvNeXtSmallClassifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="convnext_small",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=768,
            head_attribute_name="classifier",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.convnext_small(weights='DEFAULT')

    def replace_head(self, model, num_classes: int):
        if self.head_input_dim is None:
            raise ValueError("You must specify an input dimension for the model's head")
        if self.head_attribute_name is None:
            raise ValueError("You must specify the attribute name where the head is stored in")
        if self.multi_class:
            model.classifier[2] = LogitsHead(
                input_size=self.head_input_dim,
                output_size=num_classes,
                hidden_sizes=[],
            )
        else:
            model.classifier[2] = Classifier(
                input_size=self.head_input_dim,
                output_size=num_classes,
                hidden_sizes=[],
                softmax_dim=1,
            )
          
            
class ConvNeXtBaseClassifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="convnext_base",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=1024,
            head_attribute_name="classifier",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.convnext_base(weights='DEFAULT')

    def replace_head(self, model, num_classes: int):
        if self.head_input_dim is None:
            raise ValueError("You must specify an input dimension for the model's head")
        if self.head_attribute_name is None:
            raise ValueError("You must specify the attribute name where the head is stored in")
        if self.multi_class:
            model.classifier[2] = LogitsHead(
                input_size=self.head_input_dim,
                output_size=num_classes,
                hidden_sizes=[],
            )
        else:
            model.classifier[2] = Classifier(
                input_size=self.head_input_dim,
                output_size=num_classes,
                hidden_sizes=[],
                softmax_dim=1,
            )

            
class ConvNeXtLargeClassifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="convnext_large",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=1536,
            head_attribute_name="classifier",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.convnext_large(weights='DEFAULT')
 
    def replace_head(self, model, num_classes: int):
        if self.head_input_dim is None:
            raise ValueError("You must specify an input dimension for the model's head")
        if self.head_attribute_name is None:
            raise ValueError("You must specify the attribute name where the head is stored in")
        if self.multi_class:
            model.classifier[2] = LogitsHead(
                input_size=self.head_input_dim,
                output_size=num_classes,
                hidden_sizes=[],
            )
        else:
            model.classifier[2] = Classifier(
                input_size=self.head_input_dim,
                output_size=num_classes,
                hidden_sizes=[],
                softmax_dim=1,
            )
           
            
class ResNet18Classifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="resnet18",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=512,
            head_attribute_name="fc",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.resnet18(weights='DEFAULT')
            
            
class ResNet34Classifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="resnet34",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=512,
            head_attribute_name="fc",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.resnet34(weights='DEFAULT')
            
            
class ResNet50Classifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="resnet50",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=2048,
            head_attribute_name="fc",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.resnet50(weights='DEFAULT')
            
            
class ResNet101Classifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="resnet101",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=2048,
            head_attribute_name="fc",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.resnet101(weights='DEFAULT')
            
            
class ResNet152Classifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="resnet152",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=2048,
            head_attribute_name="fc",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.resnet152(weights='DEFAULT')
            
            
class ResNext5032X4DClassifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="resnext50_32x4d",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=2048,
            head_attribute_name="fc",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.resnext50_32x4d(weights='DEFAULT')
            
            
class ResNext10132X8DClassifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="resnext101_32x8d",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=2048,
            head_attribute_name="fc",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.resnext101_32x8d(weights='DEFAULT')
            
            
class ResNext10164X4DClassifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="resnext101_64x4d",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=2048,
            head_attribute_name="fc",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.resnext101_64x4d(weights='DEFAULT')
            
            
class WideResNet50Classifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="wide_resnet50_2",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=2048,
            head_attribute_name="fc",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.wide_resnet50_2(weights='DEFAULT')
            
            
class WideResNet101Classifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="wide_resnet101_2",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=2048,
            head_attribute_name="fc",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.wide_resnet101_2(weights='DEFAULT')
            
            
class SwinTClassifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="swin_t",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=768,
            head_attribute_name="head",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.swin_t(weights='DEFAULT')
            
            
class SwinSClassifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="swin_s",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=768,
            head_attribute_name="head",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.swin_s(weights='DEFAULT')
            
            
class SwinBClassifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="swin_b",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=1024,
            head_attribute_name="head",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.swin_b(weights='DEFAULT')
            
            
class SwinV2TClassifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="swin_v2_t",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=768,
            head_attribute_name="head",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.swin_v2_t(weights='DEFAULT')
            
            
class SwinV2SClassifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="swin_v2_s",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=768,
            head_attribute_name="head",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.swin_v2_s(weights='DEFAULT')
            
            
class SwinV2BClassifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="swin_v2_b",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=1024,
            head_attribute_name="head",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.swin_v2_b(weights='DEFAULT')
            
            
class ViTB16OriginalClassifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="vit_b_16_orig",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=768,
            head_attribute_name="heads",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.vit_b_16(weights='DEFAULT')
            
            
class ViTB16Classifier(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="vit_b_16",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=1576,
            head_attribute_name="heads",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return models.vit.vit_b_16(weights='DEFAULT')
    
    def replace_head(self, model, num_classes: int):
        if self.head_input_dim is None:
            raise ValueError("You must specify an input dimension for the model's head")
        if self.head_attribute_name is None:
            raise ValueError("You must specify the attribute name where the head is stored in")
        if self.multi_class:
            setattr(model, self.head_attribute_name, ViTLogitsHead(
                heads_input_dim=768,
                heads_output_dim=8,
                input_size=self.head_input_dim,
                output_size=num_classes,
                hidden_sizes=[],
            ))
        else:
            setattr(model, self.head_attribute_name, Classifier(
                input_size=self.head_input_dim,
                output_size=num_classes,
                hidden_sizes=[],
                softmax_dim=1,
            ))
            
            
class ViTB32(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="vit_b_32",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=768,
            head_attribute_name="heads",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.vit_b_32(weights='DEFAULT')
            
            
class ViTL16(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="vit_l_16",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=1024,
            head_attribute_name="heads",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.vit_l_16(weights='DEFAULT')
            
            
class ViTL32(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="vit_l_32",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=1024,
            head_attribute_name="heads",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.vit_l_32(weights='DEFAULT')
            
            
class ViTH14(AbstractClassifier):
    def __init__(
        self,
        num_classes: int = None,
        multi_class: bool = False,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        fine_tuning: bool = True,
    ):
        super().__init__(
            name="vit_h_14",
            num_classes=num_classes,
            multi_class=multi_class,
            head_input_dim=1280,
            head_attribute_name="heads",
            resume=resume,
            device=device,
            root_dir=root_dir,
            fine_tuning=fine_tuning,
        )

    def load_pretrained_model(self):
        return torchvision.models.vit_h_14(weights='DEFAULT')
