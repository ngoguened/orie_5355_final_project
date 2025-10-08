
"""
Complete DQN Implementation for Capacity-Constrained Pricing
Integration script to run all phases together
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime

# Import all phases
from phase1_demand_modeling import phase1_demand_model_training, DemandModel
from phase2_dqn_training import phase2_dqn_training, DQNTrainer, AdvancedPricingEnvironment
from phase3_hyperparameter_tuning import phase3_hyperparameter_tuning, create_final_optimized_agent
from pricing_dqn_implementation import PricingAgent

def sanitize_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [sanitize_json(i) for i in obj]
    else:
        return obj

class ProjectIntegration:
    """
    Complete integration of all phases for the pricing project
    """
    def __init__(self, project_data_path=None):
        self.project_data_path = project_data_path
        self.demand_model = None
        self.trained_agent = None
        self.final_agent = None

        # Create results directory
        os.makedirs('results', exist_ok=True)


    def load_project_data(self):
        """
        Load real ORIE5355 project data
        """
        print("Loading real ORIE5355 project data...")

        if self.project_data_path and os.path.exists(self.project_data_path):
            # Load custom project data if specified
            data = pd.read_csv(self.project_data_path)
            print(f"Loaded {len(data)} records from {self.project_data_path}")
            return data
        else:
            # Load real ORIE5355 project data
            print("Loading real ORIE5355 project training data...")
            from phase1_demand_modeling import load_project_training_data

            customer_covariates, prices, purchase_decisions = load_project_training_data()

            # Create DataFrame matching expected format (3 features instead of 5)
            data = pd.DataFrame({
                'customer_feature_1': customer_covariates[:, 0],
                'customer_feature_2': customer_covariates[:, 1],
                'customer_feature_3': customer_covariates[:, 2],
                'price': prices,
                'purchased': purchase_decisions
            })

            print(f"Loaded {len(data)} records from ORIE5355 project data")
            return data

    def run_complete_pipeline(self, quick_mode=False):
        """
        Run the complete 3-phase pipeline
        """
        print("="*60)
        print("STARTING COMPLETE DQN PRICING PIPELINE")
        print("="*60)

        start_time = datetime.now()

        # Load data
        project_data = self.load_project_data()

        # PHASE 1: Demand Model Training
        print("\n" + "="*40)
        print("PHASE 1: DEMAND MODEL TRAINING")
        print("="*40)

        if quick_mode:
            # Quick training with less data
            sample_size = min(3000, len(project_data))
            sample_data = project_data.sample(sample_size)
        else:
            sample_data = project_data

        # Train demand models
        models, sensitivity_data = phase1_demand_model_training()
        self.demand_model = models['gradient_boosting']  # Use best performing model

        print("Phase 1 completed successfully!")

        # PHASE 2: DQN Training
        print("\n" + "="*40)
        print("PHASE 2: DQN TRAINING")
        print("="*40)

        if quick_mode:
            # Reduce training episodes for quick mode
            print("Running in quick mode - reduced training episodes")

        trained_agent, training_env, training_stats = phase2_dqn_training()
        self.trained_agent = trained_agent

        print("Phase 2 completed successfully!")

        # PHASE 3: Hyperparameter Tuning (optional in quick mode)
        if not quick_mode:
            print("\n" + "="*40)
            print("PHASE 3: HYPERPARAMETER TUNING")
            print("="*40)

            optimized_agent, best_params = phase3_hyperparameter_tuning()
            self.final_agent = optimized_agent

            print("Phase 3 completed successfully!")
        else:
            print("\nSkipping hyperparameter tuning in quick mode")
            self.final_agent = self.trained_agent

        # Final evaluation and results
        print("\n" + "="*40)
        print("PIPELINE COMPLETION & RESULTS")
        print("="*40)

        end_time = datetime.now()
        total_time = end_time - start_time

        print(f"Total pipeline execution time: {total_time}")
        print("\nPipeline completed successfully!")

        # Save final results
        self.save_final_results(training_stats, total_time)

        return self.final_agent

    def save_final_results(self, training_stats, execution_time):
        """
        Save comprehensive results
        """
        print("Saving final results...")

        # Save training statistics
        if training_stats:
            stats_df = pd.DataFrame({
                'episode': range(len(training_stats['episode_rewards'])),
                'reward': training_stats['episode_rewards'],
                'revenue': training_stats['episode_revenues'],
                'epsilon': training_stats['epsilons']
            })
            stats_df.to_csv('results/final_training_statistics.csv', index=False)

        # Save execution summary
        summary = {
            'execution_time': str(execution_time),
            'timestamp': datetime.now().isoformat(),
            'final_average_revenue': np.mean(training_stats['episode_revenues'][-100:]) if training_stats else 0,
            'total_episodes_trained': len(training_stats['episode_rewards']) if training_stats else 0,
            'pipeline_stages_completed': [
                'demand_model_training',
                'dqn_training',
                'results_saved'
            ]
        }

        with open('results/execution_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print("Results saved to 'results/' directory")

    def create_submission_agent(self):
        """
        Create the final agent for project submission
        """
        if self.final_agent is None:
            raise ValueError("Must run pipeline first!")

        print("Creating submission-ready agent...")

        # The final agent is already configured for the project interface
        # Just need to set evaluation mode
        self.final_agent.epsilon = 0.0  # No exploration during evaluation

        # Save the agent
        torch.save({
            'agent_state': self.final_agent.q_network.state_dict(),
            'price_actions': self.final_agent.price_actions,
            'state_size': self.final_agent.state_size,
            'config': self.final_agent.config
        }, 'results/submission_agent.pth')

        print("Submission agent saved to 'results/submission_agent.pth'")

        return self.final_agent

    def test_agent_performance(self, n_test_episodes=100):
        """
        Test the final agent performance
        """
        if self.final_agent is None:
            raise ValueError("Must run pipeline first!")

        print(f"Testing agent performance over {n_test_episodes} episodes...")

        # Create test environment
        from phase2_dqn_training import create_customer_data_generator
        customer_generator = create_customer_data_generator()
        test_env = AdvancedPricingEnvironment(self.demand_model, customer_generator)

        # Test performance
        test_revenues = []
        test_inventories_sold = []

        # Set agent to evaluation mode
        original_epsilon = self.final_agent.epsilon
        self.final_agent.epsilon = 0.0

        for episode in range(n_test_episodes):
            customer_features = test_env.reset()

            for step in range(20):  # 20 customers per episode
                # Get state
                state = self.final_agent.get_state(
                    customer_features,
                    test_env.inventory,
                    test_env.get_time_until_replenish()
                )

                # Get action (no exploration)
                action_idx = self.final_agent.select_action(state, test_env.inventory)
                price = self.final_agent.price_actions[action_idx]

                # Take step
                reward, done, info = test_env.step(price)

                if done or test_env.customers_served >= 20:
                    break

                customer_features = test_env._get_current_customer_features()

            test_revenues.append(test_env.total_revenue)
            test_inventories_sold.append(12 - test_env.inventory)

        # Restore original epsilon
        self.final_agent.epsilon = original_epsilon

        # Calculate performance metrics
        avg_revenue = np.mean(test_revenues)
        std_revenue = np.std(test_revenues)
        avg_inventory_sold = np.mean(test_inventories_sold)

        print(f"\nTest Performance Results:")
        print(f"Average Revenue: ${avg_revenue:.2f} ± ${std_revenue:.2f}")
        print(f"Average Inventory Sold: {avg_inventory_sold:.1f}/12")
        print(f"Inventory Utilization: {avg_inventory_sold/12*100:.1f}%")

        # Save test results
        test_results = {
            'avg_revenue': avg_revenue,
            'std_revenue': std_revenue,
            'avg_inventory_sold': avg_inventory_sold,
            'inventory_utilization': avg_inventory_sold/12,
            'test_episodes': n_test_episodes,
            'all_revenues': test_revenues
        }

        test_results_sanitized = sanitize_json(test_results)
        with open('results/test_performance.json', 'w') as f:
            json.dump(test_results_sanitized, f, indent=2)

        return test_results

def main():
    """
    Main execution function
    """
    print("DQN Pricing Agent - Complete Implementation")
    print("Choose execution mode:")
    print("1. Quick mode (faster, less training)")
    print("2. Full mode (complete pipeline with tuning)")

    # For automatic execution, use quick mode
    mode = input("Enter choice (1 or 2), or press Enter for quick mode: ").strip()
    quick_mode = mode != '2'

    if quick_mode:
        print("Running in quick mode...")
    else:
        print("Running in full mode...")

    # Initialize integration
    integration = ProjectIntegration()

    try:
        # Run complete pipeline
        final_agent = integration.run_complete_pipeline(quick_mode=quick_mode)

        # Create submission agent
        submission_agent = integration.create_submission_agent()

        # Test performance
        test_results = integration.test_agent_performance(50 if quick_mode else 100)

        print("\n" + "="*60)
        print("SUCCESS: Complete pipeline executed successfully!")
        print("="*60)
        print("\nFiles created:")
        print("- results/submission_agent.pth (your final agent)")
        print("- results/final_training_statistics.csv")
        print("- results/execution_summary.json")
        print("- results/test_performance.json")
        print("\nYour agent is ready for submission!")

    except Exception as e:
        print(f"\nERROR: Pipeline execution failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
