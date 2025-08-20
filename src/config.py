import os

class MPDDConfig:
    def __init__(self, elderly_data_path=None, young_data_path=None):
        # Dataset paths (USER MUST SET THESE)
        self.elderly_data_path = elderly_data_path
        self.young_data_path = young_data_path
        
        # Feature dimensions for different modalities
        self.feature_dims = {
            'mfcc': 64,
            'wav2vec': 512,
            'opensmile': 6373,
            'resnet': 1000,
            'densenet': 1024,
            'openface': 709
        }
        
        # Training parameters
        self.batch_size = 8
        self.learning_rate = 1e-4
        self.num_epochs = 10
        self.max_length = 5
        self.dropout_rate = 0.1
        
        # Architecture parameters
        self.vision_embed_dim = 768
        self.audio_embed_dim = 512
        self.text_embed_dim = 1024
        self.fusion_dim = 256
        self.hidden_dim = 512
        self.num_attention_heads = 8
        self.num_transformer_layers = 2
        
        # Classification parameters
        self.binary_classes = 2
        self.ternary_classes = 3
        self.quinary_classes = 5
        
        # Model settings
        self.use_layer_norm = True
    
    def get_data_path(self, dataset_name):
        """Get correct data path"""
        if dataset_name == 'elderly':
            if self.elderly_data_path is None:
                raise ValueError("Please set elderly_data_path in config")
            return self.elderly_data_path
        elif dataset_name == 'young':
            if self.young_data_path is None:
                raise ValueError("Please set young_data_path in config")
            return self.young_data_path
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
