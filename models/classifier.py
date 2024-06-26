import torch.nn as nn


class Classifier(nn.Module):
    """
    This module takes the outputs of a feature extractor and runs them through a
    classification network that consists of linear layers and a final (log)softmax
    classification layer.
    
    Params:
        input_size:        size of the input features
        output_size:       size of the target features (usually the numer of classes)
        hidden_sizes:      size of the hidden layers
        softmax_dim:       dimension that contains the classification scores
        log_softmax:       true if a LogSoftmax layer should be used instead of a Softmax layer
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list[int] = [256],
        softmax_dim:int = 2,
        log_softmax: bool = True,
    ):
        super(Classifier, self).__init__()
        
        # Hyperparameters
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.log_softmax = log_softmax
        
        # Layers
        layer_dims = [input_size] + hidden_sizes + [output_size]
        self.fcs = nn.Sequential(*[nn.Linear(dim_in, dim_out) for (dim_in, dim_out) in zip(layer_dims[:-1], layer_dims[1:])])
        self.softmax = nn.LogSoftmax(dim=softmax_dim) if log_softmax else nn.Softmax(dim=softmax_dim)
        
    def forward(self, x):
        """
        Run a tensor through the model
        
        Params:
            x: the tensor
        """
        return self.softmax(self.fcs(x))
    

class LogitsHead(nn.Module):
    """
    This module takes the outputs of a feature extractor and runs them through a
    classification network that consists of linear layers. It does not contain a 
    final softmax layer.
    
    Params:
        input_size:        size of the input features
        output_size:       size of the target features (usually the numer of classes)
        hidden_sizes:      size of the hidden layers
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list[int] = [256],
    ):
        super(LogitsHead, self).__init__()
        
        # Hyperparameters
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        
        # Layers
        layer_dims = [input_size] + hidden_sizes + [output_size]
        self.fcs = nn.Sequential(*[nn.Linear(dim_in, dim_out) for (dim_in, dim_out) in zip(layer_dims[:-1], layer_dims[1:])])
        
    def forward(self, x):
        """
        Run a tensor through the model
        
        Params:
            x: the tensor
        """
        return self.fcs(x)


class ViTLogitsHead(nn.Module):
    """
    This module takes the outputs of a feature extractor and runs them through a
    classification network that consists of linear layers. It does not contain a 
    final softmax layer.
    
    Params:
        heads_input_dim:   input dimension of the vit head
        heads_output_dim:  output dimension of the vit head
        input_size:        size of the input features
        output_size:       size of the target features (usually the numer of classes)
        hidden_sizes:      size of the hidden layers
    """
    def __init__(
        self,
        heads_input_dim: int,
        heads_output_dim: int, 
        input_size: int,
        output_size: int,
        hidden_sizes: list[int] = [256],
    ):
        super(ViTLogitsHead, self).__init__()

        # Layers
        self.reducer = nn.Sequential(nn.Linear(heads_input_dim, heads_output_dim), nn.Flatten(start_dim=1))
        self.logits_head = LogitsHead(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)
        
    def forward(self, x):
        """
        Run a tensor through the model
        
        Params:
            x: the tensor
        """
        x = self.reducer(x)
        x = self.logits_head(x)
        return x