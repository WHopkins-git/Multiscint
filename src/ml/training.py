"""
Model training utilities
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, Callable
from tqdm import tqdm
import time


class ModelTrainer:
    """
    Unified trainer for all models

    Supports:
        - Standard cross-entropy loss
        - Physics-informed loss (for PINN models)
        - Mixed precision training
        - Early stopping
        - Learning rate scheduling
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        use_amp: bool = False
    ):
        """
        Parameters:
            model: PyTorch model
            device: 'cuda' or 'cpu'
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            use_amp: Use automatic mixed precision
        """
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )

        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def train_epoch(
        self,
        train_loader: DataLoader,
        loss_fn: Optional[Callable] = None,
        is_physics_informed: bool = False
    ) -> Dict[str, float]:
        """
        Train for one epoch

        Parameters:
            train_loader: Training data loader
            loss_fn: Custom loss function (default: cross-entropy)
            is_physics_informed: Whether model is physics-informed

        Returns:
            Dictionary with loss and accuracy
        """
        self.model.train()

        total_loss = 0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc='Training', leave=False)

        for batch_idx, batch_data in enumerate(progress_bar):
            # Unpack batch (format depends on dataset)
            if len(batch_data) == 3:
                waveforms, labels, metadata = batch_data
                waveforms = waveforms.to(self.device)
                labels = labels.to(self.device)
            else:
                waveforms, labels = batch_data
                waveforms = waveforms.to(self.device)
                labels = labels.to(self.device)
                metadata = None

            self.optimizer.zero_grad()

            # Forward pass with optional AMP
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    if is_physics_informed and hasattr(self.model, 'compute_loss'):
                        # Physics-informed model
                        amplitudes = metadata['amplitude'].to(self.device) if metadata is not None else \
                            torch.max(waveforms, dim=1)[0]
                        loss, loss_components = self.model.compute_loss(waveforms, labels, amplitudes)
                    else:
                        # Standard model
                        logits = self.model(waveforms)
                        loss = nn.functional.cross_entropy(logits, labels) if loss_fn is None else loss_fn(logits, labels)

                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                # Without AMP
                if is_physics_informed and hasattr(self.model, 'compute_loss'):
                    amplitudes = metadata['amplitude'].to(self.device) if metadata is not None else \
                        torch.max(waveforms, dim=1)[0]
                    loss, loss_components = self.model.compute_loss(waveforms, labels, amplitudes)
                    logits = self.model(waveforms)
                else:
                    logits = self.model(waveforms)
                    loss = nn.functional.cross_entropy(logits, labels) if loss_fn is None else loss_fn(logits, labels)

                loss.backward()
                self.optimizer.step()

            # Get predictions
            if is_physics_informed and not hasattr(self.model, 'compute_loss'):
                # Get logits again for accuracy calculation
                with torch.no_grad():
                    logits = self.model(waveforms)

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            total_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100 * correct / total
            })

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total

        return {'loss': avg_loss, 'accuracy': accuracy}

    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        loss_fn: Optional[Callable] = None
    ) -> Dict[str, float]:
        """
        Evaluate model

        Parameters:
            val_loader: Validation data loader
            loss_fn: Custom loss function

        Returns:
            Dictionary with loss and accuracy
        """
        self.model.eval()

        total_loss = 0
        correct = 0
        total = 0

        for batch_data in val_loader:
            # Unpack batch
            if len(batch_data) == 3:
                waveforms, labels, _ = batch_data
            else:
                waveforms, labels = batch_data

            waveforms = waveforms.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            logits = self.model(waveforms)
            loss = nn.functional.cross_entropy(logits, labels) if loss_fn is None else loss_fn(logits, labels)

            # Predictions
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total

        return {'loss': avg_loss, 'accuracy': accuracy}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 15,
        loss_fn: Optional[Callable] = None,
        is_physics_informed: bool = False,
        verbose: bool = True
    ) -> Dict:
        """
        Full training loop

        Parameters:
            train_loader: Training data
            val_loader: Validation data
            epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            loss_fn: Custom loss function
            is_physics_informed: Whether using physics-informed model
            verbose: Print progress

        Returns:
            Training history
        """
        best_val_acc = 0
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, loss_fn, is_physics_informed)

            # Validate
            val_metrics = self.evaluate(val_loader, loss_fn)

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])

            # Learning rate scheduling
            self.scheduler.step(val_metrics['accuracy'])

            # Early stopping
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            epoch_time = time.time() - start_time

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} ({epoch_time:.1f}s)")
                print(f"  Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
                print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
                print(f"  Best Val Acc: {best_val_acc:.2f}%")

            # Early stopping
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return self.history

    def save_checkpoint(self, filepath: str):
        """Save training checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }, filepath)

    def load_checkpoint(self, filepath: str):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
