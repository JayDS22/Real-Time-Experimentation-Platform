import numpy as np
from scipy import stats
import math
from typing import Dict, List, Tuple, Any, Optional

class PowerAnalysis:
    """
    Statistical power analysis for experiment design
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def sample_size_two_proportions(self, p1: float, p2: float, 
                                  alpha: float = 0.05, power: float = 0.8,
                                  two_sided: bool = True) -> Dict[str, Any]:
        """
        Calculate sample size for comparing two proportions
        """
        
        # Effect size
        effect_size = abs(p2 - p1)
        
        # Pooled proportion
        p_pooled = (p1 + p2) / 2
        
        # Z-values
        if two_sided:
            z_alpha = stats.norm.ppf(1 - alpha/2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)
        
        z_beta = stats.norm.ppf(power)
        
        # Sample size calculation
        numerator = (z_alpha * math.sqrt(2 * p_pooled * (1 - p_pooled)) + 
                    z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))**2
        
        denominator = effect_size**2
        
        n_per_group = math.ceil(numerator / denominator) if denominator > 0 else float('inf')
        
        return {
            'n_per_group': n_per_group,
            'total_n': n_per_group * 2,
            'effect_size': effect_size,
            'power': power,
            'alpha': alpha,
            'p1': p1,
            'p2': p2,
            'two_sided': two_sided
        }
    
    def sample_size_two_means(self, mu1: float, mu2: float, sigma: float,
                            alpha: float = 0.05, power: float = 0.8,
                            two_sided: bool = True) -> Dict[str, Any]:
        """
        Calculate sample size for comparing two means
        """
        
        # Effect size (Cohen's d)
        effect_size = abs(mu2 - mu1) / sigma
        
        # Z-values
        if two_sided:
            z_alpha = stats.norm.ppf(1 - alpha/2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)
        
        z_beta = stats.norm.ppf(power)
        
        # Sample size calculation
        n_per_group = math.ceil(2 * ((z_alpha + z_beta) / effect_size)**2)
        
        return {
            'n_per_group': n_per_group,
            'total_n': n_per_group * 2,
            'effect_size': effect_size,
            'cohens_d': effect_size,
            'power': power,
            'alpha': alpha,
            'mu1': mu1,
            'mu2': mu2,
            'sigma': sigma,
            'two_sided': two_sided
        }
    
    def power_two_proportions(self, n_per_group: int, p1: float, p2: float,
                            alpha: float = 0.05, two_sided: bool = True) -> float:
        """
        Calculate power for two-proportion test given sample size
        """
        
        effect_size = abs(p2 - p1)
        p_pooled = (p1 + p2) / 2
        
        # Standard error under null
        se_null = math.sqrt(2 * p_pooled * (1 - p_pooled) / n_per_group)
        
        # Standard error under alternative
        se_alt = math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n_per_group)
        
        # Critical value
        if two_sided:
            z_alpha = stats.norm.ppf(1 - alpha/2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)
        
        # Power calculation
        z_beta = (effect_size / se_alt) - (z_alpha * se_null / se_alt)
        power = stats.norm.cdf(z_beta)
        
        return power
    
    def minimum_detectable_effect(self, n_per_group: int, baseline_rate: float,
                                 alpha: float = 0.05, power: float = 0.8,
                                 two_sided: bool = True) -> Dict[str, Any]:
        """
        Calculate minimum detectable effect given sample size
        """
        
        # Z-values
        if two_sided:
            z_alpha = stats.norm.ppf(1 - alpha/2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)
        
        z_beta = stats.norm.ppf(power)
        
        # For proportions, solve for p2 given p1 (baseline_rate)
        p1 = baseline_rate
        
        # Quadratic equation coefficients for solving MDE
        # This is simplified - full solution involves solving quadratic equation
        se_factor = math.sqrt(2 * p1 * (1 - p1) / n_per_group)
        
        # Approximate MDE
        mde_absolute = (z_alpha + z_beta) * se_factor
        mde_relative = mde_absolute / p1 if p1 > 0 else 0
        
        return {
            'mde_absolute': mde_absolute,
            'mde_relative': mde_relative,
            'mde_relative_percent': mde_relative * 100,
            'baseline_rate': baseline_rate,
            'n_per_group': n_per_group,
            'alpha': alpha,
            'power': power
        }
