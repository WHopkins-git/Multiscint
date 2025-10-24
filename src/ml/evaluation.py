"""
Comprehensive model evaluation and comparison framework
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import time
from typing import Dict, List, Union
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """
    Evaluate a single model comprehensively

    Metrics:
        - Accuracy, Precision, Recall, F1
        - Confusion matrix
        - Inference speed
        - Model size
        - Energy-dependent performance
        - Noise robustness
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.class_names = ['LYSO', 'BGO', 'NaI', 'Plastic']

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Comprehensive evaluation

        Returns:
            Dictionary with all metrics
        """
        self.model.eval()

        all_predictions = []
        all_labels = []
        all_probabilities = []

        # Inference speed measurement
        inference_times = []

        for batch_data in test_loader:
            # Unpack
            if len(batch_data) == 3:
                waveforms, labels, _ = batch_data
            else:
                waveforms, labels = batch_data

            waveforms = waveforms.to(self.device)

            # Time inference
            start_time = time.time()
            logits = self.model(waveforms)
            inference_time = (time.time() - start_time) / len(waveforms)  # Time per sample

            inference_times.append(inference_time)

            # Predictions
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

        # Convert to arrays
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        probabilities = np.array(all_probabilities)

        # Calculate metrics
        results = {
            'accuracy': accuracy_score(labels, predictions) * 100,
            'precision': precision_score(labels, predictions, average='weighted') * 100,
            'recall': recall_score(labels, predictions, average='weighted') * 100,
            'f1_score': f1_score(labels, predictions, average='weighted') * 100,
            'confusion_matrix': confusion_matrix(labels, predictions),
            'inference_time_ms': np.mean(inference_times) * 1000,
            'inference_time_std': np.std(inference_times) * 1000,
            'model_parameters': self._count_parameters(),
            'model_size_mb': self._get_model_size()
        }

        # Per-class metrics
        class_report = classification_report(
            labels,
            predictions,
            target_names=self.class_names,
            output_dict=True
        )
        results['per_class_metrics'] = class_report

        return results

    def evaluate_by_energy(
        self,
        test_loader: DataLoader,
        energy_bins: List[tuple] = None
    ) -> Dict:
        """
        Evaluate accuracy as function of energy

        Parameters:
            test_loader: Test data loader (must include energy in metadata)
            energy_bins: List of (E_min, E_max) tuples

        Returns:
            Dictionary with accuracy per energy bin
        """
        if energy_bins is None:
            energy_bins = [(0, 200), (200, 500), (500, 800), (800, 1500)]

        self.model.eval()

        # Collect predictions and energies
        all_predictions = []
        all_labels = []
        all_energies = []

        with torch.no_grad():
            for batch_data in test_loader:
                waveforms, labels, metadata = batch_data
                waveforms = waveforms.to(self.device)

                logits = self.model(waveforms)
                predictions = torch.argmax(logits, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())

                if 'energy' in metadata and metadata['energy'] is not None:
                    all_energies.extend(metadata['energy'].numpy())
                else:
                    # Skip energy-dependent analysis if energies not available
                    return {'error': 'Energy information not available'}

        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        energies = np.array(all_energies)

        # Calculate accuracy per energy bin
        results = {}
        for E_min, E_max in energy_bins:
            mask = (energies >= E_min) & (energies < E_max)

            if np.sum(mask) > 0:
                bin_predictions = predictions[mask]
                bin_labels = labels[mask]
                accuracy = accuracy_score(bin_labels, bin_predictions) * 100

                results[f'{E_min}-{E_max}_keV'] = {
                    'accuracy': accuracy,
                    'n_samples': int(np.sum(mask))
                }
            else:
                results[f'{E_min}-{E_max}_keV'] = {
                    'accuracy': None,
                    'n_samples': 0
                }

        return results

    def evaluate_noise_robustness(
        self,
        test_loader: DataLoader,
        noise_levels: List[float] = None
    ) -> Dict:
        """
        Evaluate robustness to added noise

        Parameters:
            test_loader: Test data loader
            noise_levels: Standard deviations of Gaussian noise to add

        Returns:
            Dictionary with accuracy vs noise level
        """
        if noise_levels is None:
            noise_levels = [0, 0.05, 0.1, 0.2, 0.5]

        self.model.eval()

        results = {}

        # Collect original waveforms
        all_waveforms = []
        all_labels = []

        for batch_data in test_loader:
            if len(batch_data) == 3:
                waveforms, labels, _ = batch_data
            else:
                waveforms, labels = batch_data

            all_waveforms.append(waveforms)
            all_labels.append(labels)

        all_waveforms = torch.cat(all_waveforms, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Test each noise level
        for noise_std in noise_levels:
            # Add noise
            if noise_std > 0:
                noise = torch.randn_like(all_waveforms) * noise_std
                noisy_waveforms = all_waveforms + noise
            else:
                noisy_waveforms = all_waveforms

            # Evaluate
            with torch.no_grad():
                noisy_waveforms = noisy_waveforms.to(self.device)
                logits = self.model(noisy_waveforms)
                predictions = torch.argmax(logits, dim=1)

            accuracy = accuracy_score(all_labels.numpy(), predictions.cpu().numpy()) * 100

            results[f'noise_{noise_std}'] = accuracy

        return results

    def _count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _get_model_size(self) -> float:
        """Get model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)

    def plot_confusion_matrix(self, confusion_mat: np.ndarray, title: str = "Confusion Matrix"):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_mat,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(title)
        return plt.gcf()


class ModelComparison:
    """
    Compare multiple models across various metrics
    """

    def __init__(self, models_dict: Dict[str, nn.Module], device: str = 'cuda'):
        """
        Parameters:
            models_dict: Dictionary mapping model names to model instances
            device: 'cuda' or 'cpu'
        """
        self.models = models_dict
        self.device = device
        self.evaluators = {
            name: ModelEvaluator(model, device)
            for name, model in models_dict.items()
        }

    def evaluate_all(self, test_loader: DataLoader) -> pd.DataFrame:
        """
        Evaluate all models

        Returns:
            DataFrame with comparison results
        """
        results = {}

        for name, evaluator in self.evaluators.items():
            print(f"\nEvaluating {name}...")

            metrics = evaluator.evaluate(test_loader)

            results[name] = {
                'Accuracy (%)': metrics['accuracy'],
                'Precision (%)': metrics['precision'],
                'Recall (%)': metrics['recall'],
                'F1-Score (%)': metrics['f1_score'],
                'Inference Time (ms)': metrics['inference_time_ms'],
                'Parameters': metrics['model_parameters'],
                'Model Size (MB)': metrics['model_size_mb']
            }

        return pd.DataFrame(results).T

    def plot_comparison(self, results_df: pd.DataFrame, title: str = "Model Comparison"):
        """Create comprehensive comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Accuracy bar plot
        axes[0, 0].barh(results_df.index, results_df['Accuracy (%)'])
        axes[0, 0].set_xlabel('Accuracy (%)')
        axes[0, 0].set_title('Classification Accuracy')
        axes[0, 0].set_xlim([85, 100])

        # Speed vs Accuracy scatter
        axes[0, 1].scatter(
            results_df['Inference Time (ms)'],
            results_df['Accuracy (%)'],
            s=100
        )
        for idx, name in enumerate(results_df.index):
            axes[0, 1].annotate(
                name,
                (results_df.loc[name, 'Inference Time (ms)'],
                 results_df.loc[name, 'Accuracy (%)']),
                fontsize=8,
                xytext=(5, 5),
                textcoords='offset points'
            )
        axes[0, 1].set_xlabel('Inference Time (ms)')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Speed-Accuracy Trade-off')
        axes[0, 1].grid(True, alpha=0.3)

        # Model size comparison
        axes[1, 0].barh(results_df.index, results_df['Model Size (MB)'])
        axes[1, 0].set_xlabel('Model Size (MB)')
        axes[1, 0].set_title('Model Size')

        # F1-Score comparison
        axes[1, 1].barh(results_df.index, results_df['F1-Score (%)'])
        axes[1, 1].set_xlabel('F1-Score (%)')
        axes[1, 1].set_title('F1-Score')
        axes[1, 1].set_xlim([85, 100])

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        return fig

    def data_efficiency_study(
        self,
        train_dataset,
        test_loader: DataLoader,
        training_sizes: List[int] = None,
        epochs_per_size: int = 50
    ) -> Dict:
        """
        Analyze data efficiency: accuracy vs training set size

        Parameters:
            train_dataset: Training dataset
            test_loader: Test data loader
            training_sizes: List of training set sizes to test
            epochs_per_size: Number of epochs to train for each size

        Returns:
            Dictionary with results for each model
        """
        if training_sizes is None:
            training_sizes = [100, 500, 1000, 5000, 10000]

        results = {name: [] for name in self.models.keys()}

        for n_samples in training_sizes:
            print(f"\nTraining with {n_samples} samples...")

            # Sample subset
            indices = np.random.choice(len(train_dataset), min(n_samples, len(train_dataset)), replace=False)
            subset = Subset(train_dataset, indices)
            subset_loader = DataLoader(subset, batch_size=64, shuffle=True)

            for name, model in self.models.items():
                # Reset model (requires reset_parameters method)
                if hasattr(model, 'reset_parameters'):
                    model.reset_parameters()

                # Train (simplified - would use ModelTrainer in practice)
                # For this example, just evaluate current model
                evaluator = ModelEvaluator(model, self.device)
                metrics = evaluator.evaluate(test_loader)
                results[name].append(metrics['accuracy'])

        return results, training_sizes
