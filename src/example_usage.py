"""
Example usage of MPDD Depression Detection System

This script demonstrates how to use the MPDD system for training and evaluation.
"""

import torch
import os
from src.config import MPDDConfig
from src.utils import run_experiment, validate_experiment_config

def main():
    """
    Main function demonstrating MPDD usage
    """
    print("🚀 MPDD: Multimodal Personalized Depression Detection")
    print("=" * 60)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize configuration
    config = MPDDConfig()
    
    # IMPORTANT: Set your dataset paths here
    # Update these paths to point to your actual MPDD datasets
    config.elderly_data_path = "/path/to/your/MPDD-Elderly"
    config.young_data_path = "/path/to/your/MPDD-Young"
    
    print(f"Elderly dataset path: {config.elderly_data_path}")
    print(f"Young dataset path: {config.young_data_path}")
    
    # Example experiments to run
    experiments = [
        # Format: (dataset_name, window_size, task_type, audio_feature, visual_feature)
        ("elderly", "5s", "binary", "wav2vec", "resnet"),
        ("elderly", "5s", "ternary", "wav2vec", "resnet"),
        ("elderly", "1s", "binary", "wav2vec", "openface"),
        ("young", "5s", "binary", "wav2vec", "openface"),
        ("young", "5s", "ternary", "wav2vec", "openface"),
    ]
    
    results = []
    
    for i, (dataset_name, window_size, task_type, audio_feature, visual_feature) in enumerate(experiments):
        print(f"\n🔬 Experiment {i+1}/{len(experiments)}")
        
        # Validate configuration first
        is_valid, message = validate_experiment_config(
            dataset_name, window_size, task_type, audio_feature, visual_feature
        )
        
        if not is_valid:
            print(f"❌ Invalid configuration: {message}")
            continue
        
        # Check if dataset path exists
        data_path = config.get_data_path(dataset_name)
        if not os.path.exists(data_path):
            print(f"❌ Dataset not found at: {data_path}")
            print("Please update the dataset paths in this script")
            continue
        
        # Run experiment
        result = run_experiment(
            dataset_name=dataset_name,
            window_size=window_size,
            task_type=task_type,
            audio_feature=audio_feature,
            visual_feature=visual_feature,
            config=config,
            device=device
        )
        
        if result:
            results.append(result)
            print(f"✅ Experiment {i+1} completed successfully!")
        else:
            print(f"❌ Experiment {i+1} failed!")
    
    # Print summary of results
    print(f"\n🎉 EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Completed experiments: {len(results)}/{len(experiments)}")
    
    if results:
        print(f"\n📊 RESULTS:")
        print(f"{'Config':<40} {'Accuracy':<10} {'F1-Score':<10}")
        print("-" * 60)
        
        for result in results:
            config_str = f"{result['dataset']}-{result['window_size']}-{result['task_type']}"
            metrics = result['final_metrics']
            print(f"{config_str:<40} {metrics['accuracy']:<10.4f} {metrics['f1_weighted']:<10.4f}")
        
        # Find best result
        best_result = max(results, key=lambda x: x['final_metrics']['f1_weighted'])
        print(f"\n🏆 BEST RESULT:")
        print(f"Configuration: {best_result['dataset']}-{best_result['window_size']}-{best_result['task_type']}")
        print(f"F1-Score: {best_result['final_metrics']['f1_weighted']:.4f}")
        print(f"Accuracy: {best_result['final_metrics']['accuracy']:.4f}")

def quick_test():
    """
    Quick test with minimal configuration
    """
    print("🧪 Quick Test Mode")
    print("=" * 30)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = MPDDConfig()
    
    # Test with smaller parameters for quick testing
    config.batch_size = 4
    config.num_epochs = 2
    config.max_length = 3
    
    # Update your dataset path here
    config.elderly_data_path = "/path/to/your/MPDD-Elderly"
    
    if not os.path.exists(config.elderly_data_path):
        print("❌ Please update the dataset path in this script")
        return
    
    # Run a single quick experiment
    result = run_experiment(
        dataset_name="elderly",
        window_size="5s",
        task_type="binary",
        audio_feature="wav2vec",
        visual_feature="resnet",
        config=config,
        device=device
    )
    
    if result:
        print("✅ Quick test passed!")
    else:
        print("❌ Quick test failed!")

if __name__ == "__main__":
    # Choose which mode to run
    mode = "main"  # Change to "quick" for quick test
    
    if mode == "quick":
        quick_test()
    else:
        main()
