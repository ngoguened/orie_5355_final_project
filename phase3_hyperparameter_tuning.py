
import torch
import numpy as np
import pandas as pd
from itertools import product
import json
from datetime import datetime
import optuna
from phase2_dqn_training import DQNTrainer, AdvancedPricingEnvironment, create_customer_data_generator
from pricing_dqn_implementation import PricingAgent
from phase1_demand_modeling import DemandModel
import warnings
warnings.filterwarnings('ignore')

class HyperparameterTuner:
    """
    Comprehensive hyperparameter tuning for DQN pricing agent
    """
    def __init__(self, demand_model, base_config=None):
        self.demand_model = demand_model
        self.customer_generator = create_customer_data_generator()

        # Base configuration
        self.base_config = base_config or {
            'customer_feature_dim': 5,
            'price_actions': [10, 15, 20, 25, 30, 35, 40, 45, 50],
            'training_episodes': 800,  # Reduced for faster tuning
            'eval_episodes': 100
        }

        # Results storage
        self.tuning_results = []

    def grid_search_tuning(self):
        """
        Grid search over key hyperparameters
        """
        print("=== PHASE 3: GRID SEARCH HYPERPARAMETER TUNING ===")

        # Define hyperparameter grid
        param_grid = {
            'learning_rate': [0.0005, 0.001, 0.002],
            'gamma': [0.9, 0.95, 0.99],
            'epsilon_decay': [0.99, 0.995, 0.999],
            'batch_size': [32, 64, 128],
            'hidden_sizes': [
                [64, 64],
                [128, 128, 64],
                [256, 128, 64]
            ],
            'target_update_freq': [50, 100, 200]
        }

        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        print(f"Testing {len(combinations)} hyperparameter combinations...")

        best_score = -float('inf')
        best_params = None

        for i, combination in enumerate(combinations[:20]):  # Limit to first 20 for demo
            params = dict(zip(param_names, combination))
            print(f"\nTesting combination {i+1}/20: {params}")

            try:
                score = self._evaluate_hyperparameters(params)

                result = {
                    'combination': i+1,
                    'params': params,
                    'score': score,
                    'timestamp': datetime.now().isoformat()
                }

                self.tuning_results.append(result)

                if score > best_score:
                    best_score = score
                    best_params = params

                print(f"Score: {score:.2f}")

            except Exception as e:
                print(f"Error in combination {i+1}: {str(e)}")
                continue

        print(f"\nBest hyperparameters: {best_params}")
        print(f"Best score: {best_score:.2f}")

        return best_params, best_score

    def bayesian_optimization_tuning(self, n_trials=50):
        """
        Bayesian optimization using Optuna
        """
        print("=== PHASE 3: BAYESIAN OPTIMIZATION TUNING ===")

        def objective(trial):
            # Define hyperparameter search space
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'gamma': trial.suggest_float('gamma', 0.85, 0.99),
                'epsilon_decay': trial.suggest_float('epsilon_decay', 0.99, 0.999),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
                'hidden_sizes': trial.suggest_categorical('hidden_sizes', [
                    [64, 64],
                    [128, 128, 64],
                    [256, 128, 64],
                    [128, 256, 128]
                ]),
                'target_update_freq': trial.suggest_int('target_update_freq', 50, 200, step=25)
            }

            try:
                score = self._evaluate_hyperparameters(params)
                return score
            except Exception as e:
                print(f"Trial failed: {str(e)}")
                return -1000  # Return very low score for failed trials

        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best value: {study.best_value:.2f}")
        print(f"Best params: {study.best_params}")

        return study.best_params, study.best_value

    def _evaluate_hyperparameters(self, params):
        """
        Evaluate a set of hyperparameters
        """
        # Create environment
        env = AdvancedPricingEnvironment(self.demand_model, self.customer_generator)

        # Create agent with specified hyperparameters
        agent_config = {
            'learning_rate': params['learning_rate'],
            'gamma': params['gamma'],
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay': params['epsilon_decay'],
            'batch_size': params['batch_size'],
            'memory_size': 5000,  # Reduced for faster training
            'target_update_freq': params['target_update_freq'],
            'hidden_sizes': params['hidden_sizes']
        }

        agent = PricingAgent(
            self.base_config['customer_feature_dim'],
            self.base_config['price_actions'],
            agent_config
        )

        # Create trainer
        training_config = {
            'num_episodes': self.base_config['training_episodes'],
            'eval_frequency': 200,
            'print_frequency': 1000,  # Reduce printing
            'early_stopping_patience': 200,
            'target_revenue': 160
        }

        trainer = DQNTrainer(agent, env, training_config)

        # Train agent
        training_stats = trainer.train()

        # Evaluate performance (average revenue over last episodes)
        eval_window = min(100, len(training_stats['episode_revenues']))
        avg_revenue = np.mean(training_stats['episode_revenues'][-eval_window:])

        return avg_revenue

    def architecture_search(self):
        """
        Specific search for neural network architecture
        """
        print("=== PHASE 3: NEURAL ARCHITECTURE SEARCH ===")

        architectures = [
            [32, 32],
            [64, 64],
            [128, 128],
            [64, 64, 32],
            [128, 128, 64],
            [256, 128, 64],
            [128, 256, 128],
            [512, 256, 128],
            [64, 128, 256, 128],
            [128, 256, 512, 256, 128]
        ]

        results = []

        for i, architecture in enumerate(architectures):
            print(f"\nTesting architecture {i+1}/{len(architectures)}: {architecture}")

            params = {
                'learning_rate': 0.001,  # Use reasonable defaults
                'gamma': 0.95,
                'epsilon_decay': 0.995,
                'batch_size': 64,
                'hidden_sizes': architecture,
                'target_update_freq': 100
            }

            try:
                score = self._evaluate_hyperparameters(params)
                results.append({
                    'architecture': architecture,
                    'score': score,
                    'parameters': sum(self._count_parameters(architecture))
                })
                print(f"Score: {score:.2f}, Parameters: {results[-1]['parameters']}")

            except Exception as e:
                print(f"Architecture {architecture} failed: {str(e)}")
                continue

        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)

        print("\nArchitecture Search Results:")
        for i, result in enumerate(results[:5]):
            print(f"{i+1}. {result['architecture']} - Score: {result['score']:.2f}")

        return results

    def _count_parameters(self, hidden_sizes):
        """
        Estimate number of parameters in network
        """
        input_size = self.base_config['customer_feature_dim'] + 3  # +3 for additional state features
        output_size = len(self.base_config['price_actions'])

        param_count = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            param_count.append(prev_size * hidden_size + hidden_size)  # weights + biases
            prev_size = hidden_size

        # Output layer
        param_count.append(prev_size * output_size + output_size)

        return param_count

    def learning_rate_schedule_search(self):
        """
        Search for optimal learning rate schedules
        """
        print("=== PHASE 3: LEARNING RATE SCHEDULE SEARCH ===")

        schedules = [
            {'type': 'constant', 'lr': 0.001},
            {'type': 'constant', 'lr': 0.0005},
            {'type': 'step_decay', 'initial_lr': 0.001, 'decay_factor': 0.9, 'decay_steps': 200},
            {'type': 'exponential_decay', 'initial_lr': 0.001, 'decay_rate': 0.95},
            {'type': 'cosine_annealing', 'max_lr': 0.001, 'min_lr': 0.0001}
        ]

        results = []

        for i, schedule in enumerate(schedules):
            print(f"\nTesting schedule {i+1}/{len(schedules)}: {schedule}")

            # Note: This would require modifying the agent to support different schedules
            # For now, we'll just test different constant learning rates
            if schedule['type'] == 'constant':
                params = {
                    'learning_rate': schedule['lr'],
                    'gamma': 0.95,
                    'epsilon_decay': 0.995,
                    'batch_size': 64,
                    'hidden_sizes': [128, 128, 64],
                    'target_update_freq': 100
                }

                try:
                    score = self._evaluate_hyperparameters(params)
                    results.append({
                        'schedule': schedule,
                        'score': score
                    })
                    print(f"Score: {score:.2f}")

                except Exception as e:
                    print(f"Schedule {schedule} failed: {str(e)}")
                    continue

        return results

    def save_results(self, filename='hyperparameter_tuning_results.json'):
        """
        Save tuning results to file
        """
        with open(filename, 'w') as f:
            json.dump(self.tuning_results, f, indent=2)
        print(f"Results saved to {filename}")

def create_final_optimized_agent(best_params, customer_feature_dim, price_actions):
    """
    Create the final optimized agent with best hyperparameters
    """
    print("=== CREATING FINAL OPTIMIZED AGENT ===")

    # Combine best parameters with any remaining defaults
    final_config = {
        'learning_rate': best_params.get('learning_rate', 0.001),
        'gamma': best_params.get('gamma', 0.95),
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': best_params.get('epsilon_decay', 0.995),
        'batch_size': best_params.get('batch_size', 64),
        'memory_size': 15000,  # Larger memory for final agent
        'target_update_freq': best_params.get('target_update_freq', 100),
        'hidden_sizes': best_params.get('hidden_sizes', [128, 128, 64])
    }

    agent = PricingAgent(customer_feature_dim, price_actions, final_config)

    print("Final agent configuration:")
    for key, value in final_config.items():
        print(f"  {key}: {value}")

    return agent

def phase3_hyperparameter_tuning():
    """
    Complete Phase 3: Hyperparameter Tuning Pipeline
    """
    print("=== PHASE 3: HYPERPARAMETER TUNING PIPELINE ===")

    # Step 1: Load demand model
    print("Step 1: Loading demand model...")
    demand_model = DemandModel(model_type='gradient_boosting')

    try:
        demand_model.load_model('demand_model_gradient_boosting.pkl')
    except:
        # Create simple model if not available
        from phase1_demand_modeling import simulate_training_data
        customer_covariates, prices, purchase_decisions = simulate_training_data(3000)
        demand_model.fit(customer_covariates, prices, purchase_decisions)

    # Step 2: Initialize tuner
    print("Step 2: Initializing hyperparameter tuner...")
    tuner = HyperparameterTuner(demand_model)

    # Step 3: Run different tuning approaches
    print("Step 3: Running hyperparameter tuning...")

    # Architecture search (fastest)
    arch_results = tuner.architecture_search()
    best_architecture = arch_results[0]['architecture'] if arch_results else [128, 128, 64]

    # Grid search (subset)
    print("\nRunning limited grid search...")
    grid_best_params, grid_best_score = tuner.grid_search_tuning()

    # Bayesian optimization (if Optuna is available)
    try:
        print("\nRunning Bayesian optimization...")
        bayes_best_params, bayes_best_score = tuner.bayesian_optimization_tuning(n_trials=20)

        # Choose best approach
        if bayes_best_score > grid_best_score:
            final_best_params = bayes_best_params
            final_best_score = bayes_best_score
            print(f"Bayesian optimization won with score: {bayes_best_score:.2f}")
        else:
            final_best_params = grid_best_params
            final_best_score = grid_best_score
            print(f"Grid search won with score: {grid_best_score:.2f}")

    except ImportError:
        print("Optuna not available, using grid search results")
        final_best_params = grid_best_params
        final_best_score = grid_best_score

    # Step 4: Create final optimized agent
    print("Step 4: Creating final optimized agent...")
    final_agent = create_final_optimized_agent(
        final_best_params,
        tuner.base_config['customer_feature_dim'],
        tuner.base_config['price_actions']
    )

    # Step 5: Save results
    print("Step 5: Saving tuning results...")
    tuner.save_results('phase3_tuning_results.json')

    # Save final parameters
    with open('final_best_hyperparameters.json', 'w') as f:
        json.dump({
            'best_params': final_best_params,
            'best_score': final_best_score,
            'best_architecture': best_architecture,
            'tuning_timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print("Phase 3 hyperparameter tuning completed!")
    print(f"Best configuration score: {final_best_score:.2f}")

    return final_agent, final_best_params

# Example execution
if __name__ == "__main__":
    optimized_agent, best_hyperparams = phase3_hyperparameter_tuning()
