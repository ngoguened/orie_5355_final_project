
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pandas as pd
from phase1_demand_modeling import DemandModel
import matplotlib.pyplot as plt
import joblib

# Import the DQN components from previous implementation
from pricing_dqn_implementation import PricingDQN, ReplayBuffer, PricingAgent

class AdvancedPricingEnvironment:
    """
    Enhanced pricing environment that uses trained demand model
    """
    def __init__(self, demand_model, customer_data_generator=None):
        self.demand_model = demand_model
        self.customer_data_generator = customer_data_generator
        self.reset()

        # Environment statistics tracking
        self.episode_stats = {
            'revenues': [],
            'inventories_sold': [],
            'avg_prices': [],
            'purchase_rates': []
        }

    def reset(self):
        """Reset environment for new episode (batch of 20 customers)"""
        self.inventory = 12
        self.customers_served = 0
        self.total_revenue = 0
        self.prices_set = []
        self.purchases_made = []
        self.current_batch_customers = self._generate_customer_batch()
        return self._get_current_customer_features()

    def _generate_customer_batch(self):
        """Generate batch of 20 customers"""
        if self.customer_data_generator:
            return self.customer_data_generator(20)
        else:
            # Default: random customer generation
            return np.random.randn(20, 5)  # 20 customers, 5 features each

    def _get_current_customer_features(self):
        """Get features for current customer"""
        if self.customers_served < len(self.current_batch_customers):
            return self.current_batch_customers[self.customers_served]
        return None

    def step(self, price):
        """
        Execute one step in environment
        Returns: reward, done, info
        """
        if self.customers_served >= 20:
            return 0, True, {"error": "Episode already finished"}

        # Get current customer features
        customer_features = self._get_current_customer_features()

        # Predict purchase probability using demand model
        purchase_prob = self.demand_model.predict_proba(
            customer_features.reshape(1, -1), 
            [price]
        )[0]

        # Simulate purchase decision
        purchased = np.random.random() < purchase_prob

        # Calculate reward
        reward = 0
        if purchased and self.inventory > 0:
            reward = price  # Revenue as immediate reward
            self.inventory -= 1
            self.total_revenue += price
            self.purchases_made.append(1)
        else:
            self.purchases_made.append(0)

        self.prices_set.append(price)
        self.customers_served += 1

        # Check if episode is done
        done = self.customers_served >= 20

        # Additional reward shaping for end of episode
        if done:
            # Inventory efficiency bonus
            inventory_sold = 12 - self.inventory
            efficiency_bonus = (inventory_sold / 12) * 50  # Scale bonus
            reward += efficiency_bonus

            # Record episode statistics
            self.episode_stats['revenues'].append(self.total_revenue)
            self.episode_stats['inventories_sold'].append(inventory_sold)
            self.episode_stats['avg_prices'].append(np.mean(self.prices_set))
            self.episode_stats['purchase_rates'].append(np.mean(self.purchases_made))

        # Info for debugging
        info = {
            'inventory': self.inventory,
            'customers_served': self.customers_served,
            'total_revenue': self.total_revenue,
            'purchase_prob': purchase_prob,
            'purchased': purchased
        }

        return reward, done, info

    def get_time_until_replenish(self):
        """Get customers remaining in current batch"""
        return 20 - self.customers_served

    def get_stats(self, last_n_episodes=100):
        """Get training statistics"""
        if not self.episode_stats['revenues']:
            return {}

        n = min(last_n_episodes, len(self.episode_stats['revenues']))
        return {
            'avg_revenue': np.mean(self.episode_stats['revenues'][-n:]),
            'avg_inventory_sold': np.mean(self.episode_stats['inventories_sold'][-n:]),
            'avg_price': np.mean(self.episode_stats['avg_prices'][-n:]),
            'avg_purchase_rate': np.mean(self.episode_stats['purchase_rates'][-n:])
        }

class DQNTrainer:
    """
    Training manager for DQN agent
    """
    def __init__(self, agent, environment, config=None):
        self.agent = agent
        self.env = environment

        # Default training configuration
        default_config = {
            'num_episodes': 2000,
            'max_steps_per_episode': 20,
            'eval_frequency': 100,
            'save_frequency': 500,
            'print_frequency': 100,
            'early_stopping_patience': 300,
            'target_revenue': 200  # Target average revenue for early stopping
        }

        if config:
            default_config.update(config)
        self.config = default_config

        # Training tracking
        self.training_stats = {
            'episode_rewards': [],
            'episode_revenues': [],
            'avg_rewards': [],
            'avg_revenues': [],
            'epsilons': [],
            'losses': []
        }

    def train(self):
        """
        Main training loop
        """
        print("=== PHASE 2: DQN TRAINING ===")
        print(f"Training for {self.config['num_episodes']} episodes...")

        best_avg_revenue = 0
        patience_counter = 0

        for episode in range(self.config['num_episodes']):
            # Reset environment
            customer_features = self.env.reset()
            total_reward = 0
            episode_experiences = []

            # Run episode
            for step in range(self.config['max_steps_per_episode']):
                # Get current state
                state = self.agent.get_state(
                    customer_features,
                    self.env.inventory,
                    self.env.get_time_until_replenish()
                )

                # Select action
                action_idx = self.agent.select_action(state, self.env.inventory)
                price = self.agent.price_actions[action_idx]

                # Take step
                reward, done, info = self.env.step(price)
                total_reward += reward

                # Get next state
                next_customer_features = None
                next_state = None

                if not done and self.env.customers_served < 20:
                    next_customer_features = self.env._get_current_customer_features()
                    if next_customer_features is not None:
                        next_state = self.agent.get_state(
                            next_customer_features,
                            self.env.inventory,
                            self.env.get_time_until_replenish()
                        )

                # Store experience
                if next_state is not None:
                    experience = (state, action_idx, reward, next_state, done)
                    episode_experiences.append(experience)

                if done:
                    break

                customer_features = next_customer_features

            # Add experiences to replay buffer
            for exp in episode_experiences:
                self.agent.remember(*exp)

            # Train agent (multiple times per episode for better learning)
            for _ in range(min(5, len(episode_experiences))):
                self.agent.replay()

            # Record statistics
            self.training_stats['episode_rewards'].append(total_reward)
            self.training_stats['episode_revenues'].append(self.env.total_revenue)
            self.training_stats['epsilons'].append(self.agent.epsilon)

            # Calculate moving averages
            window = 100
            if episode >= window - 1:
                avg_reward = np.mean(self.training_stats['episode_rewards'][-window:])
                avg_revenue = np.mean(self.training_stats['episode_revenues'][-window:])
                self.training_stats['avg_rewards'].append(avg_reward)
                self.training_stats['avg_revenues'].append(avg_revenue)

            # Print progress
            if episode % self.config['print_frequency'] == 0:
                stats = self.env.get_stats(100)
                print(f"Episode {episode}:")
                print(f"  Total Reward: {total_reward:.2f}")
                print(f"  Revenue: {self.env.total_revenue:.2f}")
                print(f"  Inventory Sold: {12 - self.env.inventory}/12")
                print(f"  Epsilon: {self.agent.epsilon:.3f}")
                if stats:
                    print(f"  Avg Revenue (last 100): {stats['avg_revenue']:.2f}")
                    print(f"  Avg Purchase Rate: {stats['avg_purchase_rate']:.3f}")

            # Evaluation and early stopping
            if episode % self.config['eval_frequency'] == 0 and episode > 0:
                avg_revenue = np.mean(self.training_stats['episode_revenues'][-100:])

                if avg_revenue > best_avg_revenue:
                    best_avg_revenue = avg_revenue
                    patience_counter = 0
                    # Save best model
                    self.save_agent(f"best_dqn_agent_episode_{episode}.pth")
                else:
                    patience_counter += self.config['eval_frequency']

                # Early stopping
                if (patience_counter >= self.config['early_stopping_patience'] and 
                    avg_revenue >= self.config['target_revenue']):
                    print(f"Early stopping at episode {episode}")
                    print(f"Best average revenue: {best_avg_revenue:.2f}")
                    break

            # Save checkpoint
            if episode % self.config['save_frequency'] == 0 and episode > 0:
                self.save_agent(f"dqn_agent_checkpoint_{episode}.pth")

        print("Training completed!")
        return self.training_stats

    def save_agent(self, filepath):
        """Save trained agent"""
        torch.save({
            'q_network_state_dict': self.agent.q_network.state_dict(),
            'target_network_state_dict': self.agent.target_network.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'config': self.agent.config,
            'price_actions': self.agent.price_actions,
            'state_size': self.agent.state_size,
            'action_size': self.agent.action_size,
            'epsilon': self.agent.epsilon,
            'steps_done': self.agent.steps_done
        }, filepath)
        print(f"Agent saved to {filepath}")

    def load_agent(self, filepath, agent):
        """Load trained agent"""
        checkpoint = torch.load(filepath)
        agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.epsilon = checkpoint['epsilon']
        agent.steps_done = checkpoint['steps_done']
        print(f"Agent loaded from {filepath}")
        return agent

def create_customer_data_generator():
    """
    Create customer data generator based on your project data
    Replace this with actual data loading logic
    """
    def generate_customers(n_customers):
        # Simulate diverse customer types
        customer_types = np.random.choice(['low_value', 'medium_value', 'high_value'], 
                                        n_customers, p=[0.4, 0.4, 0.2])
        customers = []

        for customer_type in customer_types:
            if customer_type == 'low_value':
                features = np.random.normal(-0.5, 0.8, 5)
            elif customer_type == 'medium_value':
                features = np.random.normal(0, 0.6, 5)
            else:  # high_value
                features = np.random.normal(1, 0.4, 5)

            customers.append(features)

        return np.array(customers)

    return generate_customers

def phase2_dqn_training():
    """
    Complete Phase 2: DQN Training Pipeline
    """
    print("=== PHASE 2: DQN TRAINING SETUP ===")

    # Step 1: Load trained demand model
    print("Step 1: Loading trained demand model...")
    demand_model = DemandModel(model_type='gradient_boosting')

    # For demonstration, create a simple demand model if file doesn't exist
    try:
        demand_model.load_model('demand_model_gradient_boosting.pkl')
        print("Loaded existing demand model")
    except:
        print("Creating new demand model for demonstration...")
        # Create simple demand model
        from phase1_demand_modeling import simulate_training_data
        customer_covariates, prices, purchase_decisions = simulate_training_data(5000)
        demand_model.fit(customer_covariates, prices, purchase_decisions)
        demand_model.save_model('demand_model_gradient_boosting.pkl')

    # Step 2: Create training environment
    print("Step 2: Setting up training environment...")
    customer_generator = create_customer_data_generator()
    env = AdvancedPricingEnvironment(demand_model, customer_generator)

    # Step 3: Initialize DQN agent
    print("Step 3: Initializing DQN agent...")
    price_actions = [10, 15, 20, 25, 30, 35, 40, 45, 50]  # Define price range
    customer_feature_dim = 5  # Assuming 5 customer features

    # Agent configuration
    agent_config = {
        'learning_rate': 0.001,
        'gamma': 0.95,  # Slightly lower discount for shorter episodes
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay': 0.995,
        'batch_size': 64,
        'memory_size': 10000,
        'target_update_freq': 100,
        'hidden_sizes': [128, 128, 64]
    }

    agent = PricingAgent(customer_feature_dim, price_actions, agent_config)

    # Step 4: Initialize trainer
    print("Step 4: Setting up trainer...")
    training_config = {
        'num_episodes': 1500,
        'eval_frequency': 100,
        'save_frequency': 500,
        'print_frequency': 50,
        'early_stopping_patience': 300,
        'target_revenue': 180
    }

    trainer = DQNTrainer(agent, env, training_config)

    # Step 5: Run training
    print("Step 5: Starting training...")
    training_stats = trainer.train()

    # Step 6: Save final model and statistics
    print("Step 6: Saving results...")
    trainer.save_agent('final_dqn_agent.pth')

    # Save training statistics
    stats_df = pd.DataFrame({
        'episode': range(len(training_stats['episode_rewards'])),
        'reward': training_stats['episode_rewards'],
        'revenue': training_stats['episode_revenues'],
        'epsilon': training_stats['epsilons']
    })
    stats_df.to_csv('training_statistics.csv', index=False)

    print("Phase 2 training completed!")
    print(f"Final average revenue (last 100 episodes): {np.mean(training_stats['episode_revenues'][-100:]):.2f}")

    return agent, env, training_stats

# Example training execution
if __name__ == "__main__":
    trained_agent, training_env, stats = phase2_dqn_training()
