import numpy as np
from scipy import stats
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
import json

class MultiArmedBanditOptimizer:
    """
    Multi-armed bandit optimization with various algorithms
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.bandits = {}
    
    def create_bandit(self, bandit_id: str, arms: List[str], 
                     algorithm: str = 'thompson_sampling') -> str:
        """
        Create a new multi-armed bandit
        
        Args:
            bandit_id: Unique identifier
            arms: List of arm names
            algorithm: 'epsilon_greedy', 'ucb', 'thompson_sampling'
        """
        
        bandit = {
            'id': bandit_id,
            'arms': arms,
            'algorithm': algorithm,
            'created_at': datetime.now(),
            'total_pulls': 0,
            'arm_data': {arm: {'pulls': 0, 'rewards': [], 'total_reward': 0} for arm in arms}
        }
        
        # Algorithm-specific initialization
        if algorithm == 'thompson_sampling':
            # Beta distribution parameters for each arm
            bandit['beta_params'] = {arm: {'alpha': 1, 'beta': 1} for arm in arms}
        elif algorithm == 'ucb':
            # UCB doesn't need special initialization
            pass
        elif algorithm == 'epsilon_greedy':
            bandit['epsilon'] = self.config.get('epsilon', 0.1)
        
        self.bandits[bandit_id] = bandit
        
        self.logger.info(f"Created {algorithm} bandit with {len(arms)} arms")
        return bandit_id
    
    def select_arm(self, bandit_id: str) -> str:
        """
        Select an arm using the bandit algorithm
        """
        if bandit_id not in self.bandits:
            raise ValueError(f"Bandit {bandit_id} not found")
        
        bandit = self.bandits[bandit_id]
        algorithm = bandit['algorithm']
        
        if algorithm == 'thompson_sampling':
            return self._thompson_sampling_select(bandit)
        elif algorithm == 'ucb':
            return self._ucb_select(bandit)
        elif algorithm == 'epsilon_greedy':
            return self._epsilon_greedy_select(bandit)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def update_reward(self, bandit_id: str, arm: str, reward: float) -> bool:
        """
        Update bandit with observed reward
        """
        if bandit_id not in self.bandits:
            raise ValueError(f"Bandit {bandit_id} not found")
        
        bandit = self.bandits[bandit_id]
        
        if arm not in bandit['arms']:
            raise ValueError(f"Arm {arm} not found")
        
        # Update arm data
        bandit['arm_data'][arm]['pulls'] += 1
        bandit['arm_data'][arm]['rewards'].append(reward)
        bandit['arm_data'][arm]['total_reward'] += reward
        bandit['total_pulls'] += 1
        
        # Algorithm-specific updates
        if bandit['algorithm'] == 'thompson_sampling':
            self._update_thompson_sampling(bandit, arm, reward)
        
        return True
    
    def _thompson_sampling_select(self, bandit: Dict) -> str:
        """
        Thompson sampling arm selection
        """
        arm_samples = {}
        
        for arm in bandit['arms']:
            alpha = bandit['beta_params'][arm]['alpha']
            beta = bandit['beta_params'][arm]['beta']
            
            # Sample from beta distribution
            sample = np.random.beta(alpha, beta)
            arm_samples[arm] = sample
        
        # Select arm with highest sample
        selected_arm = max(arm_samples, key=arm_samples.get)
        return selected_arm
    
    def _ucb_select(self, bandit: Dict) -> str:
        """
        Upper Confidence Bound arm selection
        """
        if bandit['total_pulls'] == 0:
            # Random selection for first pull
            return np.random.choice(bandit['arms'])
        
        c = self.config.get('ucb_c', 2.0)  # Confidence parameter
        ucb_values = {}
        
        for arm in bandit['arms']:
            arm_data = bandit['arm_data'][arm]
            
            if arm_data['pulls'] == 0:
                # Infinite UCB for unpulled arms
                ucb_values[arm] = float('inf')
            else:
                mean_reward = arm_data['total_reward'] / arm_data['pulls']
                confidence = c * np.sqrt(np.log(bandit['total_pulls']) / arm_data['pulls'])
                ucb_values[arm] = mean_reward + confidence
        
        selected_arm = max(ucb_values, key=ucb_values.get)
        return selected_arm
    
    def _epsilon_greedy_select(self, bandit: Dict) -> str:
        """
        Epsilon-greedy arm selection
        """
        epsilon = bandit['epsilon']
        
        if np.random.random() < epsilon:
            # Explore: random selection
            return np.random.choice(bandit['arms'])
        else:
            # Exploit: select best arm so far
            best_arm = None
            best_mean = -float('inf')
            
            for arm in bandit['arms']:
                arm_data = bandit['arm_data'][arm]
                if arm_data['pulls'] > 0:
                    mean_reward = arm_data['total_reward'] / arm_data['pulls']
                    if mean_reward > best_mean:
                        best_mean = mean_reward
                        best_arm = arm
            
            if best_arm is None:
                # No arms pulled yet, random selection
                return np.random.choice(bandit['arms'])
            
            return best_arm
    
    def _update_thompson_sampling(self, bandit: Dict, arm: str, reward: float):
        """
        Update Thompson sampling parameters
        """
        # Assuming Bernoulli rewards (0 or 1)
        if reward > 0:
            bandit['beta_params'][arm]['alpha'] += 1
        else:
            bandit['beta_params'][arm]['beta'] += 1
    
    def get_bandit_statistics(self, bandit_id: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a bandit
        """
        if bandit_id not in self.bandits:
            raise ValueError(f"Bandit {bandit_id} not found")
        
        bandit = self.bandits[bandit_id]
        
        stats = {
            'bandit_id': bandit_id,
            'algorithm': bandit['algorithm'],
            'total_pulls': bandit['total_pulls'],
            'arm_statistics': {}
        }
        
        for arm in bandit['arms']:
            arm_data = bandit['arm_data'][arm]
            
            if arm_data['pulls'] > 0:
                rewards = np.array(arm_data['rewards'])
                arm_stats = {
                    'pulls': arm_data['pulls'],
                    'total_reward': arm_data['total_reward'],
                    'mean_reward': arm_data['total_reward'] / arm_data['pulls'],
                    'std_reward': float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0,
                    'min_reward': float(np.min(rewards)),
                    'max_reward': float(np.max(rewards)),
                    'pull_percentage': (arm_data['pulls'] / bandit['total_pulls']) * 100
                }
                
                # Confidence interval for mean
                if len(rewards) > 1:
                    ci = stats.t.interval(0.95, len(rewards)-1, 
                                        loc=np.mean(rewards), 
                                        scale=stats.sem(rewards))
                    arm_stats['confidence_interval_95'] = [float(ci[0]), float(ci[1])]
                
            else:
                arm_stats = {
                    'pulls': 0,
                    'total_reward': 0,
                    'mean_reward': 0,
                    'pull_percentage': 0
                }
            
            stats['arm_statistics'][arm] = arm_stats
        
        # Overall bandit performance
        if bandit['total_pulls'] > 0:
            # Calculate regret (difference from optimal arm)
            best_mean = max([stats['arm_statistics'][arm].get('mean_reward', 0) 
                           for arm in bandit['arms']])
            
            cumulative_regret = 0
            for arm in bandit['arms']:
                arm_data = bandit['arm_data'][arm]
                if arm_data['pulls'] > 0:
                    arm_mean = arm_data['total_reward'] / arm_data['pulls']
                    cumulative_regret += arm_data['pulls'] * (best_mean - arm_mean)
            
            stats['cumulative_regret'] = cumulative_regret
            stats['average_regret'] = cumulative_regret / bandit['total_pulls']
        
        return stats
    
    def compare_algorithms(self, arms: List[str], rewards_data: Dict[str, List[float]], 
                          num_simulations: int = 1000) -> Dict[str, Any]:
        """
        Compare different bandit algorithms on historical data
        """
        algorithms = ['epsilon_greedy', 'ucb', 'thompson_sampling']
        
        comparison_results = {
            'num_simulations': num_simulations,
            'algorithms': {},
            'summary': {}
        }
        
        for algorithm in algorithms:
            algorithm_results = []
            
            for sim in range(num_simulations):
                # Create temporary bandit
                temp_bandit_id = f"temp_{algorithm}_{sim}"
                self.create_bandit(temp_bandit_id, arms, algorithm)
                
                total_reward = 0
                total_regret = 0
                
                # Simulate pulls
                for pull in range(len(list(rewards_data.values())[0])):
                    selected_arm = self.select_arm(temp_bandit_id)
                    
                    # Get reward for this pull
                    if pull < len(rewards_data[selected_arm]):
                        reward = rewards_data[selected_arm][pull]
                        self.update_reward(temp_bandit_id, selected_arm, reward)
                        total_reward += reward
                        
                        # Calculate regret
                        best_possible_reward = max([rewards_data[arm][pull] if pull < len(rewards_data[arm]) else 0 
                                                  for arm in arms])
                        total_regret += (best_possible_reward - reward)
                
                algorithm_results.append({
                    'total_reward': total_reward,
                    'total_regret': total_regret,
                    'average_reward': total_reward / len(list(rewards_data.values())[0])
                })
                
                # Clean up temporary bandit
                del self.bandits[temp_bandit_id]
            
            # Aggregate results
            total_rewards = [result['total_reward'] for result in algorithm_results]
            total_regrets = [result['total_regret'] for result in algorithm_results]
            
            comparison_results['algorithms'][algorithm] = {
                'mean_total_reward': float(np.mean(total_rewards)),
                'std_total_reward': float(np.std(total_rewards)),
                'mean_total_regret': float(np.mean(total_regrets)),
                'std_total_regret': float(np.std(total_regrets)),
                'confidence_interval_reward': [float(x) for x in np.percentile(total_rewards, [2.5, 97.5])]
            }
        
        # Summary comparison
        best_algorithm_reward = min(comparison_results['algorithms'].keys(), 
                                  key=lambda x: comparison_results['algorithms'][x]['mean_total_regret'])
        
        comparison_results['summary'] = {
            'best_algorithm_by_regret': best_algorithm_reward,
            'performance_difference': {}
        }
        
        # Calculate relative performance
        baseline_regret = comparison_results['algorithms']['epsilon_greedy']['mean_total_regret']
        for algorithm in algorithms:
            regret = comparison_results['algorithms'][algorithm]['mean_total_regret']
            improvement = ((baseline_regret - regret) / baseline_regret) * 100 if baseline_regret > 0 else 0
            comparison_results['summary']['performance_difference'][algorithm] = f"{improvement:.1f}%"
        
        return comparison_results
