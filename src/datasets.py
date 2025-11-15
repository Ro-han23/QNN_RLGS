"""
Toy datasets for QNN training.
Includes XOR, make_moons, make_circles, and other simple classification tasks.
"""

import numpy as np
from sklearn.datasets import make_moons, make_circles, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import os


def generate_xor_dataset(n_samples: int = 200, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate XOR dataset (2-bit logical XOR).
    
    Args:
        n_samples: Number of samples to generate
        noise: Amount of noise to add to the data
    
    Returns:
        X: Input features (n_samples, 2)
        y: Binary labels (n_samples,)
    """
    np.random.seed(42)
    
    # Generate XOR pattern
    X = np.random.rand(n_samples, 2) * 2 - 1  # Values in [-1, 1]
    
    # XOR logic: y = 1 if (x0 > 0) XOR (x1 > 0), else 0
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
    
    # Add noise
    if noise > 0:
        X += np.random.randn(n_samples, 2) * noise
    
    return X, y


def load_moons_dataset(n_samples: int = 200, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load make_moons dataset from sklearn.
    
    Args:
        n_samples: Number of samples to generate
        noise: Standard deviation of Gaussian noise
    
    Returns:
        X: Input features (n_samples, 2)
        y: Binary labels (n_samples,)
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return X, y


def load_circles_dataset(n_samples: int = 200, noise: float = 0.1, factor: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load make_circles dataset from sklearn.
    
    Args:
        n_samples: Number of samples to generate
        noise: Standard deviation of Gaussian noise
        factor: Scale factor between inner and outer circle
    
    Returns:
        X: Input features (n_samples, 2)
        y: Binary labels (n_samples,)
    """
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=42)
    return X, y


def load_iris_binary_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Iris dataset for binary classification (first 2 classes).
    
    Returns:
        X: Input features (100, 4)
        y: Binary labels (100,)
    """
    iris = load_iris()
    X = iris.data[:100, :4]  # First 100 samples (2 classes)
    y = iris.target[:100]    # Binary labels (0 and 1)
    return X, y


def prepare_dataset(dataset_name: str = 'moons', 
                   test_size: float = 0.3,
                   n_samples: int = 200,
                   noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare a toy dataset for training with train/test split and standardization.
    
    Args:
        dataset_name: Name of dataset ('xor', 'moons', 'circles', 'iris')
        test_size: Fraction of data for validation
        n_samples: Number of samples to generate (for synthetic datasets)
        noise: Noise level for synthetic datasets
    
    Returns:
        X_train, X_val, y_train, y_val: Train and validation splits
    """
    # Load dataset
    if dataset_name == 'xor':
        X, y = generate_xor_dataset(n_samples=n_samples, noise=noise)
    elif dataset_name == 'moons':
        X, y = load_moons_dataset(n_samples=n_samples, noise=noise)
    elif dataset_name == 'circles':
        X, y = load_circles_dataset(n_samples=n_samples, noise=noise)
    elif dataset_name == 'iris':
        X, y = load_iris_binary_dataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from 'xor', 'moons', 'circles', 'iris'")
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    return X_train, X_val, y_train, y_val


def save_dataset(X: np.ndarray, y: np.ndarray, filename: str, data_dir: str = './data'):
    """
    Save dataset to CSV file.
    
    Args:
        X: Feature matrix
        y: Labels
        filename: Name of CSV file (without .csv extension)
        data_dir: Directory to save data
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Combine X and y
    data = np.column_stack([X, y])
    
    # Save to CSV
    filepath = os.path.join(data_dir, f"{filename}.csv")
    np.savetxt(filepath, data, delimiter=',', 
               header=','.join([f'x{i}' for i in range(X.shape[1])] + ['y']),
               comments='')
    
    print(f"Dataset saved to {filepath}")


def load_dataset_from_csv(filename: str, data_dir: str = './data') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from CSV file.
    
    Args:
        filename: Name of CSV file (without .csv extension)
        data_dir: Directory containing data
    
    Returns:
        X: Feature matrix
        y: Labels
    """
    filepath = os.path.join(data_dir, f"{filename}.csv")
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    
    return X, y


# For backward compatibility with train.py
def prepare_toy_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare toy classification dataset (defaults to moons).
    Maintains backward compatibility with existing code.
    """
    return prepare_dataset(dataset_name='moons', n_samples=200, noise=0.1)
