
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
import joblib

class NeuralDemandModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).squeeze()

class DemandModel:
    """
    Demand model to predict customer purchase probability given covariates and price
    """
    def __init__(self, model_type='logistic'):
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = None
        self.is_fitted = False

    def prepare_features(self, customer_covariates, prices):
        """
        Prepare feature matrix for model training/prediction
        """
        # Ensure inputs are numpy arrays
        if isinstance(customer_covariates, pd.DataFrame):
            customer_covariates = customer_covariates.values
        if isinstance(prices, (list, pd.Series)):
            prices = np.array(prices)

        # Reshape price if needed
        if len(prices.shape) == 1:
            prices = prices.reshape(-1, 1)

        # Combine customer features and price
        features = np.column_stack([customer_covariates, prices])
        return features

    def fit(self, customer_covariates, prices, purchase_decisions, validation_split=0.2):
        """
        Train the demand model
        """
        # Prepare features
        X = self.prepare_features(customer_covariates, prices)
        y = np.array(purchase_decisions)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Initialize model
        if self.model_type == 'logistic':
            self.model = LogisticRegression(random_state=42)
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
            )
        elif self.model_type == 'neural':
            self.model = self._create_neural_demand_model(X_train_scaled.shape[1])
            return self._train_neural_model(X_train_scaled, y_train, X_val_scaled, y_val)

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        train_pred = self.model.predict_proba(X_train_scaled)[:, 1]
        val_pred = self.model.predict_proba(X_val_scaled)[:, 1]

        print(f"Training AUC: {roc_auc_score(y_train, train_pred):.4f}")
        print(f"Validation AUC: {roc_auc_score(y_val, val_pred):.4f}")

        self.is_fitted = True
        return self

    def _create_neural_demand_model(self, input_size):
        """
        Create neural network for demand modeling
        """
        return NeuralDemandModel(input_size)

    def _train_neural_model(self, X_train, y_train, X_val, y_val, epochs=100):
        """
        Train neural demand model
        """
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)

        # Optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        # Training loop
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # Training
            self.model.train()
            optimizer.zero_grad()
            train_pred = self.model(X_train_tensor)
            train_loss = criterion(train_pred, y_train_tensor)
            train_loss.backward()
            optimizer.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val_tensor)
                val_loss = criterion(val_pred, y_val_tensor)

            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())

            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            train_pred = self.model(X_train_tensor).numpy()
            val_pred = self.model(X_val_tensor).numpy()

        print(f"Final Training AUC: {roc_auc_score(y_train, train_pred):.4f}")
        print(f"Final Validation AUC: {roc_auc_score(y_val, val_pred):.4f}")

        self.is_fitted = True
        return self

    def predict_proba(self, customer_covariates, prices):
        """
        Predict purchase probability
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Prepare features
        X = self.prepare_features(customer_covariates, prices)
        X_scaled = self.scaler.transform(X)

        if self.model_type == 'neural':
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled)
                probas = self.model(X_tensor).numpy()
            return probas
        else:
            return self.model.predict_proba(X_scaled)[:, 1]

    def save_model(self, filepath):
        """
        Save trained model
        """
        model_data = {
            'model_type': self.model_type,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }

        if self.model_type == 'neural':
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_architecture': self.model,
                'metadata': model_data
            }, filepath)
        else:
            model_data['model'] = self.model
            joblib.dump(model_data, filepath)

    def load_model(self, filepath):
        """
        Load trained model
        """
        if self.model_type == 'neural':
            checkpoint = torch.load(filepath)
            self.model = checkpoint['model_architecture']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            metadata = checkpoint['metadata']
        else:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            metadata = model_data

        self.scaler = metadata['scaler']
        self.is_fitted = metadata['is_fitted']
        return self

def simulate_training_data(n_samples=10000):
    """
    Generate synthetic training data for demonstration
    Replace this with your actual data loading
    """
    np.random.seed(42)

    # Generate customer covariates (5 features)
    customer_features = np.random.randn(n_samples, 5)

    # Generate prices (between 10 and 50)
    prices = np.random.uniform(10, 50, n_samples)

    # Generate purchase decisions based on realistic model
    # Higher customer value (sum of positive features) and lower price = higher probability
    customer_value = np.sum(np.maximum(customer_features, 0), axis=1)
    normalized_price = (prices - 10) / 40  # Normalize to 0-1

    # Logit model for purchase probability
    logit = 2 * customer_value - 3 * normalized_price - 1
    purchase_prob = 1 / (1 + np.exp(-logit))
    purchase_decisions = np.random.binomial(1, purchase_prob, n_samples)

    return customer_features, prices, purchase_decisions

# Example usage for Phase 1
def phase1_demand_model_training():
    """
    Phase 1: Complete demand model training pipeline
    """
    print("=== PHASE 1: DEMAND MODEL TRAINING ===")

    # Step 1: Load or simulate training data
    print("Step 1: Loading training data...")
    customer_covariates, prices, purchase_decisions = simulate_training_data()

    print(f"Data shape: {customer_covariates.shape[0]} samples, {customer_covariates.shape[1]} customer features")
    print(f"Purchase rate: {np.mean(purchase_decisions):.3f}")

    # Step 2: Train different demand models
    models = {}

    for model_type in ['logistic', 'gradient_boosting', 'neural']:
        print(f"\nStep 2: Training {model_type} demand model...")

        demand_model = DemandModel(model_type=model_type)
        demand_model.fit(customer_covariates, prices, purchase_decisions)

        # Save model
        model_filename = f"demand_model_{model_type}.pkl"
        if model_type == 'neural':
            model_filename = f"demand_model_{model_type}.pt"

        demand_model.save_model(model_filename)
        models[model_type] = demand_model

        print(f"Model saved as {model_filename}")

    # Step 3: Model comparison on test set
    print("\nStep 3: Model comparison...")
    test_customers, test_prices, test_purchases = simulate_training_data(1000)

    for model_name, model in models.items():
        pred_probs = model.predict_proba(test_customers, test_prices)
        auc = roc_auc_score(test_purchases, pred_probs)
        print(f"{model_name.title()} Model Test AUC: {auc:.4f}")

    # Step 4: Price sensitivity analysis
    print("\nStep 4: Price sensitivity analysis...")

    # Use best performing model (gradient boosting typically performs well)
    best_model = models['gradient_boosting']

    # Generate test customer
    test_customer = np.random.randn(1, 5)
    price_range = np.linspace(10, 50, 20)

    probabilities = []
    for price in price_range:
        prob = best_model.predict_proba(test_customer, [price])[0]
        probabilities.append(prob)

    # Create price sensitivity plot data
    sensitivity_data = pd.DataFrame({
        'price': price_range,
        'purchase_probability': probabilities
    })

    print("Price sensitivity analysis completed.")
    print(f"Price range: ${price_range[0]:.0f} - ${price_range[-1]:.0f}")
    print(f"Probability range: {min(probabilities):.3f} - {max(probabilities):.3f}")

    return models, sensitivity_data

# Run Phase 1
if __name__ == "__main__":
    models, sensitivity_data = phase1_demand_model_training()
