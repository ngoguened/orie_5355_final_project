import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pandas as pd

class PricingDQN(nn.Module):
    """
    Deep Q-Network for capacity-constrained pricing decisions
    """
    def __init__(self, state_size, action_size, hidden_sizes=[128, 128, 64]):
        super(PricingDQN, self).__init__()

        # Build network layers
        layers = []
        prev_size = state_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)  # Add dropout for regularization
            ])
            prev_size = hidden_size

        # Output layer for Q-values
        layers.append(nn.Linear(prev_size, action_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    """
    Experience replay buffer for storing transitions
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class PricingAgent:
    """
    DQN Agent for capacity-constrained pricing
    """
    def __init__(self, customer_feature_dim, price_actions, config=None):
        # Default configuration
        default_config = {
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'batch_size': 64,
            'memory_size': 10000,
            'target_update_freq': 100,
            'hidden_sizes': [128, 128, 64]
        }

        if config:
            default_config.update(config)
        self.config = default_config

        # State representation: [customer_features, inventory_level, customers_remaining_in_batch, avg_price_so_far]
        self.state_size = customer_feature_dim + 3  # +3 for inventory, customers_remaining, avg_price
        self.action_size = len(price_actions)
        self.price_actions = np.array(price_actions)

        # Initialize networks
        self.q_network = PricingDQN(self.state_size, self.action_size, self.config['hidden_sizes'])
        self.target_network = PricingDQN(self.state_size, self.action_size, self.config['hidden_sizes'])
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config['learning_rate'])

        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Experience replay
        self.memory = ReplayBuffer(self.config['memory_size'])

        # Training parameters
        self.epsilon = self.config['epsilon_start']
        self.steps_done = 0

        # Episode tracking
        self.current_inventory = 12
        self.customers_in_current_batch = 0
        self.batch_prices = []

    def get_state(self, customer_features, inventory, customers_remaining):
        """
        Construct state representation
        """
        avg_price = np.mean(self.batch_prices) if self.batch_prices else 0.0

        state = np.concatenate([
            customer_features,
            [inventory / 12.0],  # Normalize inventory
            [customers_remaining / 20.0],  # Normalize remaining customers
            [avg_price / np.max(self.price_actions)]  # Normalize average price
        ])

        return state.astype(np.float32)

    def select_action(self, state, inventory=None):
        """
        Epsilon-greedy action selection with inventory constraint
        """
        # If no inventory, must set price to discourage purchases (highest price)
        if inventory is not None and inventory <= 0:
            return len(self.price_actions) - 1  # Highest price action

        if random.random() > self.epsilon:
            # Exploitation: choose best action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return q_values.max(1)[1].item()
        else:
            # Exploration: random action
            return random.randrange(self.action_size)

    def action(self, observation):
        """
        Main action method for the project interface
        observation: (customer_covariates, sale, profits, inventories, time_until_replenish)
        """
        customer_covariates, sale, profits, inventories, time_until_replenish = observation

        # Update batch tracking
        if time_until_replenish == 19:  # New batch starting
            self.current_inventory = 12
            self.customers_in_current_batch = 0
            self.batch_prices = []

        # Get current state
        customers_remaining = time_until_replenish
        state = self.get_state(customer_covariates, inventories, customers_remaining)

        # Select action
        action_idx = self.select_action(state, inventories)
        price = self.price_actions[action_idx]

        # Track for state construction
        self.batch_prices.append(price)
        self.customers_in_current_batch += 1

        return price

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer
        """
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        """
        Train the network on a batch of experiences
        """
        if len(self.memory) < self.config['batch_size']:
            return

        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.config['batch_size'])

        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)

        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.config['gamma'] * next_q_values * ~dones)

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.config['epsilon_end']:
            self.epsilon *= self.config['epsilon_decay']

        self.steps_done += 1

        # Update target network
        if self.steps_done % self.config['target_update_freq'] == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

# Training simulation environment for testing
class PricingEnvironment:
    """
    Simulated environment for training the DQN agent
    """
    def __init__(self, demand_model=None):
        self.demand_model = demand_model  # Trained demand model from project data
        self.reset()

    def reset(self):
        self.inventory = 12
        self.customers_served = 0
        self.total_revenue = 0
        self.customer_batch = []
        return self._get_initial_state()

    def _get_initial_state(self):
        # Generate synthetic customer features for demonstration
        customer_features = np.random.randn(5)  # Assume 5 customer features
        return customer_features

    def step(self, price, customer_features):
        """
        Simulate customer purchase decision and return reward
        """
        # Simplified demand model - replace with actual trained model
        # Higher price = lower probability of purchase
        purchase_prob = max(0, 1 - price / np.max([10, 20, 30, 40, 50]))  # Example price range
        purchased = np.random.random() < purchase_prob

        reward = 0
        if purchased and self.inventory > 0:
            reward = price  # Revenue as reward
            self.inventory -= 1
            self.total_revenue += price

        self.customers_served += 1
        done = self.customers_served >= 20  # End of batch

        if done:
            # Bonus reward for efficient inventory usage
            inventory_efficiency = (12 - self.inventory) / 12
            reward += inventory_efficiency * 10  # Bonus for selling more inventory

        return reward, done

    def get_time_until_replenish(self):
        return 20 - self.customers_served

# Example usage and training loop
def train_pricing_agent():
    """
    Example training loop for the pricing agent
    """
    # Define price action space (example)
    price_actions = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    customer_feature_dim = 5  # Assume 5 customer features from data

    # Initialize agent and environment
    agent = PricingAgent(customer_feature_dim, price_actions)
    env = PricingEnvironment()

    num_episodes = 1000
    scores = []

    for episode in range(num_episodes):
        customer_features = env.reset()
        total_reward = 0

        for step in range(20):  # 20 customers per batch
            # Get state
            state = agent.get_state(
                customer_features, 
                env.inventory, 
                env.get_time_until_replenish()
            )

            # Select action
            action_idx = agent.select_action(state, env.inventory)
            price = price_actions[action_idx]

            # Take step in environment
            reward, done = env.step(price, customer_features)
            total_reward += reward

            # Get next state (next customer)
            if not done:
                next_customer_features = np.random.randn(5)  # Next customer
                next_state = agent.get_state(
                    next_customer_features,
                    env.inventory,
                    env.get_time_until_replenish()
                )
            else:
                next_state = None

            # Store experience
            if next_state is not None:
                agent.remember(state, action_idx, reward, next_state, done)

            # Train agent
            agent.replay()

            if done:
                break

            customer_features = next_customer_features

        scores.append(total_reward)

        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")

    return agent

print("DQN implementation framework created successfully!")
print("Key components:")
print("1. PricingDQN - Neural network architecture")
print("2. PricingAgent - Main agent class with action() method for project interface")
print("3. ReplayBuffer - Experience replay for stable training")
print("4. PricingEnvironment - Training environment simulation")
print("5. Training loop example")