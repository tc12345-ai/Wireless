"""
Distribution Fitting Module.

Performs statistical analysis and distribution fitting for channel data:
- Rayleigh, Rice, Gaussian distribution fitting
- Goodness-of-fit tests (K-S, Chi-square)
- Q-Q plots
- Statistical summary reports
"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class FittingResult:
    """Result of distribution fitting."""
    distribution_name: str
    parameters: Dict[str, float]
    ks_statistic: float  # Kolmogorov-Smirnov statistic
    ks_pvalue: float  # K-S test p-value
    chi2_statistic: Optional[float] = None
    chi2_pvalue: Optional[float] = None
    fitted_pdf: Optional[np.ndarray] = None
    x_values: Optional[np.ndarray] = None


@dataclass
class StatisticalSummary:
    """Statistical summary of data."""
    mean: float
    variance: float
    std: float
    skewness: float
    kurtosis: float
    min_val: float
    max_val: float
    median: float
    percentile_5: float
    percentile_95: float
    samples: int


class DistributionFitter:
    """
    Fit and test statistical distributions for channel data.
    """
    
    SUPPORTED_DISTRIBUTIONS = ['rayleigh', 'rice', 'gaussian', 'nakagami', 'lognormal']
    
    @staticmethod
    def fit_rayleigh(data: np.ndarray) -> FittingResult:
        """
        Fit Rayleigh distribution to data.
        
        The Rayleigh distribution models the magnitude of a complex
        Gaussian random variable (|X + jY| where X,Y are i.i.d. Gaussian).
        
        Args:
            data: Channel magnitude data (|h|)
            
        Returns:
            FittingResult with fitted parameters and test statistics
        """
        data = np.abs(data).flatten()
        data = data[data > 0]  # Remove zeros
        
        # Fit Rayleigh distribution
        loc, scale = stats.rayleigh.fit(data, floc=0)
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.kstest(data, 'rayleigh', args=(loc, scale))
        
        # Generate fitted PDF
        x = np.linspace(0, np.max(data) * 1.2, 100)
        pdf = stats.rayleigh.pdf(x, loc=loc, scale=scale)
        
        # Chi-square test
        chi2_stat, chi2_pval = DistributionFitter._chi2_test(
            data, lambda x: stats.rayleigh.cdf(x, loc=loc, scale=scale)
        )
        
        return FittingResult(
            distribution_name='Rayleigh',
            parameters={'loc': loc, 'scale': scale, 'sigma': scale / np.sqrt(2)},
            ks_statistic=ks_stat,
            ks_pvalue=ks_pval,
            chi2_statistic=chi2_stat,
            chi2_pvalue=chi2_pval,
            fitted_pdf=pdf,
            x_values=x
        )
    
    @staticmethod
    def fit_rice(data: np.ndarray) -> FittingResult:
        """
        Fit Rice distribution to data.
        
        The Rice distribution models the magnitude of a complex
        Gaussian with non-zero mean (LOS component).
        
        Args:
            data: Channel magnitude data
            
        Returns:
            FittingResult with fitted parameters
        """
        data = np.abs(data).flatten()
        data = data[data > 0]
        
        # Fit Rice distribution 
        # scipy.stats.rice parameterization: b = nu/sigma
        b, loc, scale = stats.rice.fit(data, floc=0)
        
        # Convert to K-factor
        # K = b^2 / 2 (power ratio)
        k_factor = b ** 2 / 2
        
        # K-S test
        ks_stat, ks_pval = stats.kstest(data, 'rice', args=(b, loc, scale))
        
        # Generate fitted PDF
        x = np.linspace(0, np.max(data) * 1.2, 100)
        pdf = stats.rice.pdf(x, b, loc=loc, scale=scale)
        
        # Chi-square test
        chi2_stat, chi2_pval = DistributionFitter._chi2_test(
            data, lambda x: stats.rice.cdf(x, b, loc=loc, scale=scale)
        )
        
        return FittingResult(
            distribution_name='Rice',
            parameters={
                'b': b, 'loc': loc, 'scale': scale,
                'K_factor': k_factor, 'K_factor_dB': 10 * np.log10(k_factor + 1e-10)
            },
            ks_statistic=ks_stat,
            ks_pvalue=ks_pval,
            chi2_statistic=chi2_stat,
            chi2_pvalue=chi2_pval,
            fitted_pdf=pdf,
            x_values=x
        )
    
    @staticmethod
    def fit_gaussian(data: np.ndarray, complex_data: bool = False) -> FittingResult:
        """
        Fit Gaussian distribution to data.
        
        Args:
            data: Data to fit
            complex_data: If True, fit to real/imag parts separately
            
        Returns:
            FittingResult with fitted parameters
        """
        if complex_data and np.iscomplexobj(data):
            # Test if real and imaginary parts are Gaussian
            real_data = data.real.flatten()
            imag_data = data.imag.flatten()
            
            mu_real, std_real = stats.norm.fit(real_data)
            mu_imag, std_imag = stats.norm.fit(imag_data)
            
            ks_real, pval_real = stats.kstest(real_data, 'norm', args=(mu_real, std_real))
            ks_imag, pval_imag = stats.kstest(imag_data, 'norm', args=(mu_imag, std_imag))
            
            return FittingResult(
                distribution_name='Complex Gaussian',
                parameters={
                    'mu_real': mu_real, 'std_real': std_real,
                    'mu_imag': mu_imag, 'std_imag': std_imag
                },
                ks_statistic=(ks_real + ks_imag) / 2,
                ks_pvalue=min(pval_real, pval_imag)
            )
        else:
            data = np.real(data).flatten()
            mu, std = stats.norm.fit(data)
            
            ks_stat, ks_pval = stats.kstest(data, 'norm', args=(mu, std))
            
            x = np.linspace(mu - 4*std, mu + 4*std, 100)
            pdf = stats.norm.pdf(x, mu, std)
            
            chi2_stat, chi2_pval = DistributionFitter._chi2_test(
                data, lambda x: stats.norm.cdf(x, mu, std)
            )
            
            return FittingResult(
                distribution_name='Gaussian',
                parameters={'mean': mu, 'std': std, 'variance': std**2},
                ks_statistic=ks_stat,
                ks_pvalue=ks_pval,
                chi2_statistic=chi2_stat,
                chi2_pvalue=chi2_pval,
                fitted_pdf=pdf,
                x_values=x
            )
    
    @staticmethod
    def fit_nakagami(data: np.ndarray) -> FittingResult:
        """
        Fit Nakagami distribution to data.
        
        Args:
            data: Channel magnitude data
            
        Returns:
            FittingResult with fitted parameters
        """
        data = np.abs(data).flatten()
        data = data[data > 0]
        
        # Fit Nakagami-m distribution
        nu, loc, scale = stats.nakagami.fit(data, floc=0)
        
        ks_stat, ks_pval = stats.kstest(data, 'nakagami', args=(nu, loc, scale))
        
        x = np.linspace(0, np.max(data) * 1.2, 100)
        pdf = stats.nakagami.pdf(x, nu, loc=loc, scale=scale)
        
        return FittingResult(
            distribution_name='Nakagami-m',
            parameters={'m': nu, 'omega': scale**2, 'loc': loc, 'scale': scale},
            ks_statistic=ks_stat,
            ks_pvalue=ks_pval,
            fitted_pdf=pdf,
            x_values=x
        )
    
    @staticmethod
    def fit_all(data: np.ndarray) -> Dict[str, FittingResult]:
        """
        Fit all supported distributions and return results.
        
        Args:
            data: Data to fit
            
        Returns:
            Dictionary of distribution name to FittingResult
        """
        results = {}
        
        # Magnitude data for fading distributions
        magnitude = np.abs(data)
        
        results['Rayleigh'] = DistributionFitter.fit_rayleigh(magnitude)
        results['Rice'] = DistributionFitter.fit_rice(magnitude)
        results['Nakagami'] = DistributionFitter.fit_nakagami(magnitude)
        results['Gaussian'] = DistributionFitter.fit_gaussian(data.real)
        
        if np.iscomplexobj(data):
            results['Complex Gaussian'] = DistributionFitter.fit_gaussian(data, complex_data=True)
        
        return results
    
    @staticmethod
    def find_best_fit(data: np.ndarray) -> Tuple[str, FittingResult]:
        """
        Find the best fitting distribution.
        
        Args:
            data: Data to fit
            
        Returns:
            Tuple of (best distribution name, FittingResult)
        """
        results = DistributionFitter.fit_all(data)
        
        # Best fit has highest p-value (fails to reject null hypothesis)
        best_name = max(results, key=lambda k: results[k].ks_pvalue)
        
        return best_name, results[best_name]
    
    @staticmethod
    def compute_summary(data: np.ndarray) -> StatisticalSummary:
        """
        Compute statistical summary of data.
        
        Args:
            data: Data array
            
        Returns:
            StatisticalSummary with computed statistics
        """
        data = np.asarray(data).flatten()
        
        # For complex data, use magnitude
        if np.iscomplexobj(data):
            data = np.abs(data)
        
        return StatisticalSummary(
            mean=np.mean(data),
            variance=np.var(data),
            std=np.std(data),
            skewness=float(stats.skew(data)),
            kurtosis=float(stats.kurtosis(data)),
            min_val=np.min(data),
            max_val=np.max(data),
            median=np.median(data),
            percentile_5=np.percentile(data, 5),
            percentile_95=np.percentile(data, 95),
            samples=len(data)
        )
    
    @staticmethod
    def generate_qq_data(data: np.ndarray, 
                        distribution: str = 'rayleigh') -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Q-Q plot data for a given distribution.
        
        Args:
            data: Observed data
            distribution: Target distribution name
            
        Returns:
            Tuple of (theoretical quantiles, sample quantiles)
        """
        data = np.abs(data).flatten()
        data = np.sort(data)
        n = len(data)
        
        # Expected probabilities
        probs = (np.arange(1, n + 1) - 0.5) / n
        
        # Theoretical quantiles
        if distribution.lower() == 'rayleigh':
            loc, scale = stats.rayleigh.fit(data, floc=0)
            theoretical = stats.rayleigh.ppf(probs, loc=loc, scale=scale)
        elif distribution.lower() == 'rice':
            b, loc, scale = stats.rice.fit(data, floc=0)
            theoretical = stats.rice.ppf(probs, b, loc=loc, scale=scale)
        elif distribution.lower() == 'gaussian' or distribution.lower() == 'normal':
            mu, std = stats.norm.fit(data)
            theoretical = stats.norm.ppf(probs, mu, std)
        else:
            # Default to Rayleigh
            loc, scale = stats.rayleigh.fit(data, floc=0)
            theoretical = stats.rayleigh.ppf(probs, loc=loc, scale=scale)
        
        return theoretical, data
    
    @staticmethod
    def _chi2_test(data: np.ndarray, cdf_func, num_bins: int = 20) -> Tuple[float, float]:
        """
        Perform chi-square goodness-of-fit test.
        
        Args:
            data: Observed data
            cdf_func: CDF function of fitted distribution
            num_bins: Number of bins for histogram
            
        Returns:
            Tuple of (chi-square statistic, p-value)
        """
        try:
            # Create histogram
            observed, bin_edges = np.histogram(data, bins=num_bins)
            
            # Compute expected counts from CDF
            expected_probs = np.diff(cdf_func(bin_edges))
            expected = expected_probs * len(data)
            
            # Avoid zero expected values
            mask = expected > 5
            if np.sum(mask) < 5:
                return np.nan, np.nan
            
            observed = observed[mask]
            expected = expected[mask]
            
            # Chi-square test
            chi2_stat = np.sum((observed - expected) ** 2 / expected)
            dof = len(observed) - 3  # 3 estimated parameters typically
            if dof <= 0:
                dof = 1
            
            chi2_pval = 1 - stats.chi2.cdf(chi2_stat, dof)
            
            return chi2_stat, chi2_pval
        except Exception:
            return np.nan, np.nan
    
    @staticmethod
    def generate_report(data: np.ndarray) -> str:
        """
        Generate a comprehensive statistical analysis report.
        
        Args:
            data: Channel data to analyze
            
        Returns:
            Formatted report string
        """
        summary = DistributionFitter.compute_summary(data)
        fits = DistributionFitter.fit_all(data)
        best_name, best_fit = DistributionFitter.find_best_fit(data)
        
        report = []
        report.append("=" * 60)
        report.append("CHANNEL STATISTICAL ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        report.append("BASIC STATISTICS")
        report.append("-" * 40)
        report.append(f"Samples:        {summary.samples}")
        report.append(f"Mean:           {summary.mean:.6f}")
        report.append(f"Std Dev:        {summary.std:.6f}")
        report.append(f"Variance:       {summary.variance:.6f}")
        report.append(f"Skewness:       {summary.skewness:.4f}")
        report.append(f"Kurtosis:       {summary.kurtosis:.4f}")
        report.append(f"Min:            {summary.min_val:.6f}")
        report.append(f"Max:            {summary.max_val:.6f}")
        report.append(f"Median:         {summary.median:.6f}")
        report.append(f"5th Percentile: {summary.percentile_5:.6f}")
        report.append(f"95th Percentile:{summary.percentile_95:.6f}")
        report.append("")
        
        report.append("DISTRIBUTION FITTING RESULTS")
        report.append("-" * 40)
        
        for name, fit in fits.items():
            report.append(f"\n{name}:")
            report.append(f"  K-S Statistic: {fit.ks_statistic:.4f}")
            report.append(f"  K-S p-value:   {fit.ks_pvalue:.4f}")
            if fit.chi2_statistic is not None and not np.isnan(fit.chi2_statistic):
                report.append(f"  Chi2 Statistic:{fit.chi2_statistic:.4f}")
                report.append(f"  Chi2 p-value:  {fit.chi2_pvalue:.4f}")
            report.append(f"  Parameters:    {fit.parameters}")
        
        report.append("")
        report.append("=" * 60)
        report.append(f"BEST FIT: {best_name}")
        report.append(f"  K-S p-value: {best_fit.ks_pvalue:.4f}")
        if best_fit.ks_pvalue > 0.05:
            report.append("  Result: PASS (p > 0.05, cannot reject fit)")
        else:
            report.append("  Result: FAIL (p <= 0.05, poor fit)")
        report.append("=" * 60)
        
        return "\n".join(report)
