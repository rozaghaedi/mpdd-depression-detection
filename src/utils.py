import torch
from torch.utils.data import DataLoader
import os
from .config import MPDDConfig
from .dataset import MPDDDataset
from .models import EnhancedMPDDModel
from .training import MPDDTrainer

def run_experiment(dataset_name, window_size, task_type, audio_feature, visual_feature, config, device):
    """
    Run a complete experiment for given configuration
    """
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {dataset_name} - {window_size} - {task_type} - {audio_feature} - {visual_feature}")
    print(f"{'='*80}")

    # Set data path based on dataset
    data_path = config.get_data_path(dataset_name)

    # Verify the dataset path exists
    if not os.path.exists(data_path):
        print(f"❌  Dataset path does not exist: {data_path}")
        return None

    # Verify essential files exist
    labels_file = os.path.join(data_path, 'Training', 'labels', 'Training_Validation_files.json')
    if not os.path.exists(labels_file):
        print(f"❌  Labels file not found: {labels_file}")
        return None

    # Verify feature directories exist
    audio_path = os.path.join(data_path, 'Training', window_size, 'Audio', audio_feature)
    visual_path = os.path.join(data_path, 'Training', window_size, 'Visual', visual_feature)

    if not os.path.exists(audio_path):
        print(f"❌  Audio feature path does not exist: {audio_path}")
        return None

    if not os.path.exists(visual_path):
        print(f"❌  Visual feature path does not exist: {visual_path}")
        return None

    # Set number of classes
    num_classes_map = {'binary': 2, 'ternary': 3, 'quinary': 5}
    num_classes = num_classes_map[task_type]

    try:
        # Create dataset
        print("Loading dataset...")
        dataset = MPDDDataset(
            data_path=data_path,
            window_size=window_size,
            audio_feature_type=audio_feature,
            visual_feature_type=visual_feature,
            task_type=task_type,
            config=config
        )

        # Check if dataset has enough samples
        if len(dataset) < 10:
            print(f"❌  Dataset too small: {len(dataset)} samples. Need at least 10 samples.")
            return None

        # Split dataset into train/validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        # Ensure minimum sizes
        if train_size < 5 or val_size < 2:
            print(f"❌  Dataset split too small: train={train_size}, val={val_size}")
            return None

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        # Determine feature dimensions based on feature types
        audio_dim = config.feature_dims.get(audio_feature, 512)
        visual_dim = config.feature_dims.get(visual_feature, 1000)

        # Adjust densenet dimension for different tracks
        if visual_feature == 'densenet' and dataset_name == 'young':
            visual_dim = 1000

        # Initialize model
        print(f"Initializing model with audio_dim={audio_dim}, visual_dim={visual_dim}...")
        model = EnhancedMPDDModel(config, num_classes=num_classes,
                                audio_input_dim=audio_dim, visual_input_dim=visual_dim)

        # Initialize trainer
        trainer = MPDDTrainer(model, config, device)

        # Train model
        trainer.train(train_loader, val_loader, config.num_epochs)

        # Plot training history
        trainer.plot_training_history()

        # Final evaluation
        print("\nFinal Evaluation:")
        val_loss, val_metrics = trainer.validate_epoch(val_loader)
        print(f"Final Validation Results:")
        print(f"  - Loss: {val_loss:.4f}")
        print(f"  - Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  - F1 (Weighted): {val_metrics['f1_weighted']:.4f}")
        print(f"  - F1 (Unweighted): {val_metrics['f1_unweighted']:.4f}")

        return {
            'dataset': dataset_name,
            'window_size': window_size,
            'task_type': task_type,
            'audio_feature': audio_feature,
            'visual_feature': visual_feature,
            'final_metrics': val_metrics,
            'model': model
        }

    except Exception as e:
        print(f"❌  Experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def validate_experiment_config(dataset_name, window_size, task_type, audio_feature, visual_feature):
    """
    Validate if experiment configuration is feasible
    """
    # Basic validation logic
    if dataset_name not in ['elderly', 'young']:
        return False, f"Invalid dataset name: {dataset_name}"
    
    if window_size not in ['1s', '5s']:
        return False, f"Invalid window size: {window_size}"
    
    if task_type not in ['binary', 'ternary', 'quinary']:
        return False, f"Invalid task type: {task_type}"
    
    # Young dataset doesn't support quinary classification
    if dataset_name == 'young' and task_type == 'quinary':
        return False, "Quinary classification not supported for young dataset"
    
    return True, "Configuration is valid"

def load_model_for_inference(model_path, config_path, device):
    """Load a trained model for inference"""
    import json
    
    # Load configuration
    with open(config_path, 'r') as f:
        saved_config = json.load(f)

    # Create config object
    config = MPDDConfig()
    
    # Determine number of classes
    task_type = saved_config['task_type']
    num_classes = {'binary': 2, 'ternary': 3, 'quinary': 5}[task_type]

    # Get feature dimensions
    audio_dim = config.feature_dims.get(saved_config['audio_feature'], 512)
    visual_dim = config.feature_dims.get(saved_config['visual_feature'], 1000)

    # Initialize model
    model = EnhancedMPDDModel(config, num_classes=num_classes,
                            audio_input_dim=audio_dim, visual_input_dim=visual_dim)

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, saved_config
