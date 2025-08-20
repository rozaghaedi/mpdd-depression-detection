import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm
from transformers import get_scheduler
import matplotlib.pyplot as plt

class MPDDTrainer:
    """
    Comprehensive trainer for MPDD models with advanced features:
    - Multi-task learning with auxiliary losses
    - Focal loss for imbalanced classes
    - Learning rate scheduling
    - Early stopping
    - Comprehensive evaluation metrics
    """

    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Import FocalLoss from models
        from .models import FocalLoss
        
        # Loss functions
        self.criterion_main = FocalLoss(alpha=1, gamma=2)
        self.criterion_aux = nn.CrossEntropyLoss()

        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )

        self.scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=100,
            num_training_steps=1000  # Will be updated based on dataset size
        )

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.early_stopping_patience = 5

    def compute_metrics(self, predictions, labels):
        """Compute comprehensive evaluation metrics"""
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()

        # Convert logits to predictions
        if predictions.ndim > 1:
            predictions = np.argmax(predictions, axis=1)

        accuracy = accuracy_score(labels, predictions)
        f1_weighted = f1_score(labels, predictions, average='weighted')
        f1_unweighted = f1_score(labels, predictions, average='macro')

        return {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_unweighted': f1_unweighted
        }

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            try:
                # Move data to device
                audio_feat = batch['audio_features'].to(self.device)
                visual_feat = batch['visual_features'].to(self.device)
                personalized_feat = batch['personalized_features'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(audio_feat, visual_feat, personalized_feat)

                # Multi-task loss computation
                main_loss = self.criterion_main(outputs['logits'], labels)
                aux_audio_loss = self.criterion_aux(outputs['aux_audio_logits'], labels)
                aux_visual_loss = self.criterion_aux(outputs['aux_visual_logits'], labels)

                # Combine losses
                total_loss_batch = main_loss + 0.3 * aux_audio_loss + 0.3 * aux_visual_loss

                # Check for NaN losses
                if torch.isnan(total_loss_batch):
                    print(f"⚠️  NaN loss detected at batch {batch_idx}, skipping...")
                    continue

                # Backward pass
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                # Statistics
                total_loss += total_loss_batch.item()
                predictions = torch.argmax(outputs['logits'], dim=1)
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)

                # Update progress bar
                pbar.set_postfix({
                    'loss': total_loss_batch.item(),
                    'acc': total_correct / total_samples
                })

            except Exception as e:
                print(f"❌  Error in training batch {batch_idx}: {str(e)}")
                continue

        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Move data to device
                    audio_feat = batch['audio_features'].to(self.device)
                    visual_feat = batch['visual_features'].to(self.device)
                    personalized_feat = batch['personalized_features'].to(self.device)
                    labels = batch['label'].to(self.device)

                    # Forward pass
                    outputs = self.model(audio_feat, visual_feat, personalized_feat)

                    # Loss computation
                    main_loss = self.criterion_main(outputs['logits'], labels)
                    aux_audio_loss = self.criterion_aux(outputs['aux_audio_logits'], labels)
                    aux_visual_loss = self.criterion_aux(outputs['aux_visual_logits'], labels)

                    total_loss_batch = main_loss + 0.3 * aux_audio_loss + 0.3 * aux_visual_loss

                    # Check for NaN losses
                    if torch.isnan(total_loss_batch):
                        print(f"⚠️  NaN loss detected in validation batch {batch_idx}, skipping...")
                        continue

                    total_loss += total_loss_batch.item()

                    # Collect predictions and labels
                    all_predictions.append(outputs['logits'])
                    all_labels.append(labels)

                    pbar.set_postfix({'val_loss': total_loss_batch.item()})

                except Exception as e:
                    print(f"❌  Error in validation batch {batch_idx}: {str(e)}")
                    continue

        # Compute metrics
        if len(all_predictions) == 0:
            print("⚠️  No valid predictions in validation, returning default metrics")
            return float('inf'), {'accuracy': 0.0, 'f1_weighted': 0.0, 'f1_unweighted': 0.0}

        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        avg_loss = total_loss / max(len(val_loader), 1)
        metrics = self.compute_metrics(all_predictions, all_labels)

        return avg_loss, metrics

    def train(self, train_loader, val_loader, num_epochs):
        """Complete training loop with early stopping"""
        print(f"Starting training for {num_epochs} epochs...")

        # Update scheduler
        self.scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=len(train_loader) * 2,
            num_training_steps=len(train_loader) * num_epochs
        )

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            # Validation
            val_loss, val_metrics = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_metrics['accuracy'])

            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"Val F1 (Weighted): {val_metrics['f1_weighted']:.4f}")
            print(f"Val F1 (Unweighted): {val_metrics['f1_unweighted']:.4f}")

            # Early stopping
            if val_metrics['f1_weighted'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1_weighted']
                self.patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
                print("✓ New best model saved!")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        print("Training completed!")

    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss plot
        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.val_losses, label='Val Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()

        # Accuracy plot
        axes[1].plot(self.train_accuracies, label='Train Acc')
        axes[1].plot(self.val_accuracies, label='Val Acc')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()

        plt.tight_layout()
        plt.show()
