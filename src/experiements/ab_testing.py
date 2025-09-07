import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime, timedelta
import uuid

class ABTestingFramework:
    """
    Comprehensive A/B testing framework with statistical rigor
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.experiments = {}
        
    def create_experiment(self, name: str, metric_type: str, 
                         variants: List[str], allocation: List[float],
                         hypothesis: str, **kwargs) -> str:
        """
        Create a new A/B test experiment
        
        Args:
            name: Experiment name
            metric_type: 'conversion', 'continuous', 'revenue'
            variants: List of variant names (control, treatment1, etc.)
            allocation: Traffic allocation for each variant
            hypothesis: Experimental hypothesis
            
        Returns:
            experiment_id: Unique identifier for the experiment
        """
        if len(variants) != len(allocation):
            raise ValueError("Number of variants must match allocation list")
        
        if not np.isclose(sum(allocation), 1.0):
            raise ValueError("Allocation must sum to 1.0")
        
        experiment_id = str(uuid.uuid4())
        
        # Calculate required sample size
        effect_size = kwargs.get('effect_size', 0.05)  # 5% relative improvement
        alpha = kwargs.get('alpha', 0.05)
        power = kwargs.get('power', 0.8)
        
        sample_size = self._calculate_sample_size(metric_type, effect_size, alpha, power)
        
        experiment = {
            'id': experiment_id,
            'name': name,
            'metric_type': metric_type,
            'variants': variants,
            'allocation': allocation,
            'hypothesis': hypothesis,
            'status': 'draft',
            'created_at': datetime.now(),
            'sample_size_per_variant': sample_size,
            'alpha': alpha,
            'power': power,
            'effect_size': effect_size,
            'data': {variant: [] for variant in variants},
            'analysis_results': {}
        }
        
        self.experiments[experiment_id] = experiment
        
        self.logger.info(f"Created experiment {name} with ID {experiment_id}")
        return experiment_id
    
    def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        experiment['status'] = 'running'
        experiment['started_at'] = datetime.now()
        
        self.logger.info(f"Started experiment {experiment['name']}")
        return True
    
    def assign_variant(self, experiment_id: str, user_id: str) -> str:
        """
        Assign a user to a variant using deterministic hashing
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        if experiment['status'] != 'running':
            raise ValueError(f"Experiment {experiment_id} is not running")
        
        # Deterministic assignment using hash
        hash_value = hash(f"{experiment_id}_{user_id}") % 10000
        cumulative_allocation = np.cumsum(experiment['allocation']) * 10000
        
        for i, threshold in enumerate(cumulative_allocation):
            if hash_value < threshold:
                assigned_variant = experiment['variants'][i]
                break
        
        return assigned_variant
    
    def record_metric(self, experiment_id: str, user_id: str, 
                     variant: str, value: float, 
                     timestamp: Optional[datetime] = None) -> bool:
        """
        Record a metric observation for the experiment
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        if variant not in experiment['variants']:
            raise ValueError(f"Variant {variant} not found in experiment")
        
        observation = {
            'user_id': user_id,
            'value': value,
            'timestamp': timestamp or datetime.now()
        }
        
        experiment['data'][variant].append(observation)
        
        # Check if we should run interim analysis
        if self._should_run_interim_analysis(experiment):
            self._run_interim_analysis(experiment_id)
        
        return True
    
    def analyze_experiment(self, experiment_id: str, 
                          test_type: str = 'ttest') -> Dict[str, Any]:
        """
        Analyze experiment results using specified statistical test
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        # Extract data for analysis
        variant_data = {}
        for variant in experiment['variants']:
            values = [obs['value'] for obs in experiment['data'][variant]]
            variant_data[variant] = np.array(values)
        
        if test_type == 'ttest':
            results = self._run_ttest_analysis(variant_data, experiment)
        elif test_type == 'chi_square':
            results = self._run_chi_square_analysis(variant_data, experiment)
        elif test_type == 'mann_whitney':
            results = self._run_mann_whitney_analysis(variant_data, experiment)
        elif test_type == 'bayesian':
            results = self._run_bayesian_analysis(variant_data, experiment)
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
        
        # Store results
        experiment['analysis_results'] = results
        experiment['last_analyzed'] = datetime.now()
        
        return results
    
    def _calculate_sample_size(self, metric_type: str, effect_size: float, 
                             alpha: float, power: float) -> int:
        """
        Calculate required sample size for experiment
        """
        if metric_type == 'conversion':
            # For conversion rate (proportion test)
            baseline_rate = self.config.get('baseline_conversion_rate', 0.1)
            
            # Two-proportion z-test sample size
            p1 = baseline_rate
            p2 = baseline_rate * (1 + effect_size)
            
            pooled_p = (p1 + p2) / 2
            
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = stats.norm.ppf(power)
            
            numerator = (z_alpha * np.sqrt(2 * pooled_p * (1 - pooled_p)) + 
                        z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))**2
            denominator = (p2 - p1)**2
            
            sample_size = int(np.ceil(numerator / denominator))
            
        elif metric_type == 'continuous':
            # For continuous metrics (t-test)
            baseline_std = self.config.get('baseline_std', 1.0)
            
            # Two-sample t-test sample size
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = stats.norm.ppf(power)
            
            sample_size = int(np.ceil(2 * (baseline_std**2) * ((z_alpha + z_beta) / effect_size)**2))
            
        else:
            # Default conservative estimate
            sample_size = 1000
        
        return max(sample_size, 100)  # Minimum sample size
    
    def _run_ttest_analysis(self, variant_data: Dict[str, np.ndarray], 
                          experiment: Dict) -> Dict[str, Any]:
        """
        Run t-test analysis for continuous metrics
        """
        control_variant = experiment['variants'][0]  # Assume first is control
        control_data = variant_data[control_variant]
        
        results = {
            'test_type': 'ttest',
            'control_variant': control_variant,
            'sample_sizes': {variant: len(data) for variant, data in variant_data.items()},
            'means': {variant: float(np.mean(data)) for variant, data in variant_data.items()},
            'std_devs': {variant: float(np.std(data, ddof=1)) for variant, data in variant_data.items()},
            'comparisons': {}
        }
        
        # Compare each treatment to control
        for variant in experiment['variants'][1:]:
            treatment_data = variant_data[variant]
            
            if len(control_data) == 0 or len(treatment_data) == 0:
                continue
                
            # Welch's t-test (unequal variances)
            t_stat, p_value = stats.ttest_ind(treatment_data, control_data, equal_var=False)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(control_data) - 1) * np.var(control_data, ddof=1) + 
                                (len(treatment_data) - 1) * np.var(treatment_data, ddof=1)) / 
                               (len(control_data) + len(treatment_data) - 2))
            
            cohens_d = (np.mean(treatment_data) - np.mean(control_data)) / pooled_std if pooled_std > 0 else 0
            
            # Confidence interval for difference in means
            se_diff = pooled_std * np.sqrt(1/len(control_data) + 1/len(treatment_data))
            df = len(control_data) + len(treatment_data) - 2
            t_critical = stats.t.ppf(1 - experiment['alpha']/2, df)
            
            mean_diff = np.mean(treatment_data) - np.mean(control_data)
            ci_lower = mean_diff - t_critical * se_diff
            ci_upper = mean_diff + t_critical * se_diff
            
            results['comparisons'][variant] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < experiment['alpha'],
                'cohens_d': float(cohens_d),
                'mean_difference': float(mean_diff),
                'confidence_interval': [float(ci_lower), float(ci_upper)],
                'degrees_of_freedom': int(df)
            }
        
        return results
    
    def _run_chi_square_analysis(self, variant_data: Dict[str, np.ndarray], 
                               experiment: Dict) -> Dict[str, Any]:
        """
        Run chi-square test for conversion rate analysis
        """
        # Convert continuous data to binary (assuming conversion data)
        conversion_data = {}
        for variant, data in variant_data.items():
            conversion_data[variant] = {
                'conversions': int(np.sum(data)),
                'total': len(data),
                'rate': np.mean(data) if len(data) > 0 else 0
            }
        
        # Chi-square test of independence
        variants = list(conversion_data.keys())
        conversions = [conversion_data[v]['conversions'] for v in variants]
        totals = [conversion_data[v]['total'] for v in variants]
        
        # Create contingency table
        non_conversions = [total - conv for total, conv in zip(totals, conversions)]
        contingency_table = np.array([conversions, non_conversions])
        
        if contingency_table.shape[1] < 2:
            return {'error': 'Need at least 2 variants for chi-square test'}
        
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        results = {
            'test_type': 'chi_square',
            'conversion_rates': {variant: conversion_data[variant]['rate'] for variant in variants},
            'sample_sizes': {variant: conversion_data[variant]['total'] for variant in variants},
            'conversions': {variant: conversion_data[variant]['conversions'] for variant in variants},
            'chi2_statistic': float(chi2_stat),
            'p_value': float(p_value),
            'degrees_of_freedom': int(dof),
            'significant': p_value < experiment['alpha']
        }
        
        return results
    
    def _run_mann_whitney_analysis(self, variant_data: Dict[str, np.ndarray], 
                                 experiment: Dict) -> Dict[str, Any]:
        """
        Run Mann-Whitney U test (non-parametric)
        """
        control_variant = experiment['variants'][0]
        control_data = variant_data[control_variant]
        
        results = {
            'test_type': 'mann_whitney',
            'control_variant': control_variant,
            'sample_sizes': {variant: len(data) for variant, data in variant_data.items()},
            'medians': {variant: float(np.median(data)) for variant, data in variant_data.items()},
            'comparisons': {}
        }
        
        for variant in experiment['variants'][1:]:
            treatment_data = variant_data[variant]
            
            if len(control_data) == 0 or len(treatment_data) == 0:
                continue
            
            # Mann-Whitney U test
            u_stat, p_value = stats.mannwhitneyu(treatment_data, control_data, 
                                               alternative='two-sided')
            
            # Effect size (rank-biserial correlation)
            r = 1 - (2 * u_stat) / (len(control_data) * len(treatment_data))
            
            results['comparisons'][variant] = {
                'u_statistic': float(u_stat),
                'p_value': float(p_value),
                'significant': p_value < experiment['alpha'],
                'rank_biserial_correlation': float(r),
                'median_difference': float(np.median(treatment_data) - np.median(control_data))
            }
        
        return results
    
    def _run_bayesian_analysis(self, variant_data: Dict[str, np.ndarray], 
                             experiment: Dict) -> Dict[str, Any]:
        """
        Run Bayesian analysis using conjugate priors
        """
        import pymc as pm
        
        results = {
            'test_type': 'bayesian',
            'sample_sizes': {variant: len(data) for variant, data in variant_data.items()},
            'posteriors': {}
        }
        
        # For each variant, compute posterior distribution
        for variant, data in variant_data.items():
            if len(data) == 0:
                continue
                
            # Assuming normal likelihood with normal prior
            data_mean = np.mean(data)
            data_std = np.std(data, ddof=1)
            n = len(data)
            
            # Conjugate normal-normal model
            prior_mean = self.config.get('prior_mean', 0)
            prior_precision = self.config.get('prior_precision', 1)
            likelihood_precision = n / (data_std**2) if data_std > 0 else 1
            
            # Posterior parameters
            posterior_precision = prior_precision + likelihood_precision
            posterior_mean = (prior_precision * prior_mean + likelihood_precision * data_mean) / posterior_precision
            posterior_std = 1 / np.sqrt(posterior_precision)
            
            # Credible interval
            credible_interval = stats.norm.interval(0.95, posterior_mean, posterior_std)
            
            results['posteriors'][variant] = {
                'mean': float(posterior_mean),
                'std': float(posterior_std),
                'credible_interval_95': [float(credible_interval[0]), float(credible_interval[1])],
                'samples': np.random.normal(posterior_mean, posterior_std, 1000).tolist()
            }
        
        # Probability that treatment is better than control
        if len(results['posteriors']) >= 2:
            control_variant = experiment['variants'][0]
            if control_variant in results['posteriors']:
                control_samples = np.array(results['posteriors'][control_variant]['samples'])
                
                for variant in experiment['variants'][1:]:
                    if variant in results['posteriors']:
                        treatment_samples = np.array(results['posteriors'][variant]['samples'])
                        prob_better = np.mean(treatment_samples > control_samples)
                        results['posteriors'][variant]['prob_better_than_control'] = float(prob_better)
        
        return results
    
    def _should_run_interim_analysis(self, experiment: Dict) -> bool:
        """
        Determine if interim analysis should be run
        """
        # Run interim analysis every 100 observations per variant
        interim_frequency = self.config.get('interim_frequency', 100)
        
        for variant in experiment['variants']:
            if len(experiment['data'][variant]) % interim_frequency == 0 and len(experiment['data'][variant]) > 0:
                return True
        
        return False
    
    def _run_interim_analysis(self, experiment_id: str):
        """
        Run interim analysis with sequential testing
        """
        # For now, just log that interim analysis would run
        # In production, would implement sequential probability ratio test (SPRT)
        # or group sequential methods with spending functions
        
        experiment = self.experiments[experiment_id]
        self.logger.info(f"Running interim analysis for experiment {experiment['name']}")
        
        # Simple implementation: check if we have strong evidence to stop early
        if len(experiment['analysis_results']) > 0:
            latest_results = experiment['analysis_results']
            
            # Example early stopping rule: p-value < 0.001 or effect size > 20%
            if 'comparisons' in latest_results:
                for variant, comparison in latest_results['comparisons'].items():
                    if comparison.get('p_value', 1) < 0.001 or abs(comparison.get('cohens_d', 0)) > 0.8:
                        self.logger.warning(f"Strong evidence detected for {variant}, consider early stopping")
