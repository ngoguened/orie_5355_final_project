"""
Data loader for ORIE5355 project data
Replaces synthetic data generation with real data from the project repository
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional

class ORIE5355DataLoader:
    """
    Data loader for the ORIE5355 project data
    """
    
    def __init__(self, data_path: str = "project_code_2024_publicshare/data/"):
        """
        Initialize the data loader
        
        Args:
            data_path: Path to the data directory
        """
        self.data_path = data_path
        self.train_data = None
        self.test_data = None
        
    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load training data from the ORIE5355 project
        
        Returns:
            customer_covariates: Customer features (n_samples, 3)
            prices: Prices offered (n_samples,)
            purchase_decisions: Purchase decisions (n_samples,)
        """
        train_file = os.path.join(self.data_path, "train_prices_decisions_2024.csv")
        
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training data file not found: {train_file}")
            
        self.train_data = pd.read_csv(train_file)
        
        # Extract customer covariates (3 features)
        customer_covariates = self.train_data[['Covariate1', 'Covariate2', 'Covariate3']].values
        
        # Extract prices
        prices = self.train_data['price_item'].values
        
        # Extract purchase decisions (convert boolean to int)
        purchase_decisions = self.train_data['item_bought'].astype(int).values
        
        print(f"Loaded training data: {len(self.train_data)} samples")
        print(f"Customer features shape: {customer_covariates.shape}")
        print(f"Purchase rate: {np.mean(purchase_decisions):.3f}")
        print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
        
        return customer_covariates, prices, purchase_decisions
    
    def load_test_data(self) -> np.ndarray:
        """
        Load test data (customer covariates only)
        
        Returns:
            test_customer_covariates: Test customer features (n_samples, 3)
        """
        test_file = os.path.join(self.data_path, "test_user_info_2024.csv")
        
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test data file not found: {test_file}")
            
        self.test_data = pd.read_csv(test_file)
        
        # Extract customer covariates (3 features)
        test_customer_covariates = self.test_data[['Covariate1', 'Covariate2', 'Covariate3']].values
        
        print(f"Loaded test data: {len(self.test_data)} samples")
        print(f"Test customer features shape: {test_customer_covariates.shape}")
        
        return test_customer_covariates
    
    def get_data_statistics(self) -> dict:
        """
        Get statistics about the loaded data
        
        Returns:
            Dictionary with data statistics
        """
        if self.train_data is None:
            self.load_training_data()
            
        stats = {
            'n_train_samples': len(self.train_data),
            'n_features': 3,
            'purchase_rate': self.train_data['item_bought'].mean(),
            'price_stats': {
                'min': self.train_data['price_item'].min(),
                'max': self.train_data['price_item'].max(),
                'mean': self.train_data['price_item'].mean(),
                'std': self.train_data['price_item'].std()
            },
            'covariate_stats': {
                'Covariate1': {
                    'min': self.train_data['Covariate1'].min(),
                    'max': self.train_data['Covariate1'].max(),
                    'mean': self.train_data['Covariate1'].mean(),
                    'std': self.train_data['Covariate1'].std()
                },
                'Covariate2': {
                    'min': self.train_data['Covariate2'].min(),
                    'max': self.train_data['Covariate2'].max(),
                    'mean': self.train_data['Covariate2'].mean(),
                    'std': self.train_data['Covariate2'].std()
                },
                'Covariate3': {
                    'min': self.train_data['Covariate3'].min(),
                    'max': self.train_data['Covariate3'].max(),
                    'mean': self.train_data['Covariate3'].mean(),
                    'std': self.train_data['Covariate3'].std()
                }
            }
        }
        
        return stats
    
    def create_validation_split(self, validation_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create train/validation split from the training data
        
        Args:
            validation_size: Fraction of data to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_val, prices_train, prices_val, y_train, y_val
        """
        if self.train_data is None:
            self.load_training_data()
            
        from sklearn.model_selection import train_test_split
        
        # Get features and targets
        X = self.train_data[['Covariate1', 'Covariate2', 'Covariate3']].values
        prices = self.train_data['price_item'].values
        y = self.train_data['item_bought'].astype(int).values
        
        # Create stratified split to maintain purchase rate balance
        X_train, X_val, prices_train, prices_val, y_train, y_val = train_test_split(
            X, prices, y, 
            test_size=validation_size, 
            random_state=random_state, 
            stratify=y
        )
        
        return X_train, X_val, prices_train, prices_val, y_train, y_val


def load_project_training_data(data_path: str = "project_code_2024_publicshare/data/") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to load training data
    Replaces the simulate_training_data() function in the original code
    
    Args:
        data_path: Path to the data directory
        
    Returns:
        customer_covariates: Customer features (n_samples, 3)
        prices: Prices offered (n_samples,)
        purchase_decisions: Purchase decisions (n_samples,)
    """
    loader = ORIE5355DataLoader(data_path)
    return loader.load_training_data()


def load_project_test_data(data_path: str = "project_code_2024_publicshare/data/") -> np.ndarray:
    """
    Convenience function to load test data
    
    Args:
        data_path: Path to the data directory
        
    Returns:
        test_customer_covariates: Test customer features (n_samples, 3)
    """
    loader = ORIE5355DataLoader(data_path)
    return loader.load_test_data()


# Example usage and testing
if __name__ == "__main__":
    # Test the data loader
    loader = ORIE5355DataLoader()
    
    # Load training data
    print("Loading training data...")
    customer_covariates, prices, purchase_decisions = loader.load_training_data()
    
    # Load test data
    print("\nLoading test data...")
    test_covariates = loader.load_test_data()
    
    # Get statistics
    print("\nData statistics:")
    stats = loader.get_data_statistics()
    print(f"Training samples: {stats['n_train_samples']}")
    print(f"Features: {stats['n_features']}")
    print(f"Purchase rate: {stats['purchase_rate']:.3f}")
    print(f"Price range: ${stats['price_stats']['min']:.2f} - ${stats['price_stats']['max']:.2f}")
    
    # Test validation split
    print("\nTesting validation split...")
    X_train, X_val, prices_train, prices_val, y_train, y_val = loader.create_validation_split()
    print(f"Train set: {len(X_train)} samples, purchase rate: {np.mean(y_train):.3f}")
    print(f"Validation set: {len(X_val)} samples, purchase rate: {np.mean(y_val):.3f}")