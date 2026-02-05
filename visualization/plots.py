"""
Channel Visualization Module.

Provides plotting functions for channel analysis:
- Power Delay Profile (PDP)
- Autocorrelation function
- Frequency response
- Channel magnitude
- Doppler spectrum
- Distribution histograms
- Q-Q plots
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Optional, Tuple, List

# Use try/except for flexible import
try:
    from analysis.statistics import ChannelStatistics, PDPResult, AutocorrelationResult
    from analysis.distribution_fitting import DistributionFitter, FittingResult
except ImportError:
    from ..analysis.statistics import ChannelStatistics, PDPResult, AutocorrelationResult
    from ..analysis.distribution_fitting import DistributionFitter, FittingResult



class ChannelPlotter:
    """
    Plotting utilities for channel analysis visualization.
    """
    
    # Default style settings
    STYLE = {
        'figure.figsize': (10, 6),
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 1.5,
        'font.size': 10,
    }
    
    def __init__(self, style: dict = None):
        """
        Initialize plotter with optional custom style.
        
        Args:
            style: Custom matplotlib style dictionary
        """
        self.style = {**self.STYLE, **(style or {})}
    
    def apply_style(self):
        """Apply plotting style settings."""
        for key, value in self.style.items():
            plt.rcParams[key] = value
    
    @staticmethod
    def plot_pdp(pdp_result: PDPResult, ax: plt.Axes = None, 
                 title: str = "Power Delay Profile") -> plt.Axes:
        """
        Plot Power Delay Profile.
        
        Args:
            pdp_result: PDPResult from ChannelStatistics
            ax: Matplotlib axes (created if None)
            title: Plot title
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Convert delays to microseconds for readability
        delays_us = pdp_result.delays * 1e6
        power_db = 10 * np.log10(pdp_result.power + 1e-10)
        
        # Stem plot for PDP
        markerline, stemlines, baseline = ax.stem(
            delays_us, power_db, linefmt='b-', markerfmt='bo', basefmt='k-'
        )
        plt.setp(stemlines, linewidth=2)
        plt.setp(markerline, markersize=8)
        
        ax.set_xlabel('Delay (μs)')
        ax.set_ylabel('Power (dB)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = (f"RMS Delay: {pdp_result.rms_delay_spread*1e6:.3f} μs\n"
                     f"Mean Delay: {pdp_result.mean_delay*1e6:.3f} μs\n"
                     f"Max Delay: {pdp_result.max_delay*1e6:.3f} μs")
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)
        
        return ax
    
    @staticmethod
    def plot_autocorrelation(autocorr_result: AutocorrelationResult, 
                            ax: plt.Axes = None,
                            title: str = "Temporal Autocorrelation") -> plt.Axes:
        """
        Plot autocorrelation function.
        
        Args:
            autocorr_result: AutocorrelationResult from ChannelStatistics
            ax: Matplotlib axes
            title: Plot title
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Time lags in milliseconds
        time_lags_ms = autocorr_result.time_lags * 1e3
        
        ax.plot(time_lags_ms, np.abs(autocorr_result.autocorr), 'b-', linewidth=2)
        ax.axhline(y=0.5, color='r', linestyle='--', label='50% Correlation')
        ax.axvline(x=autocorr_result.coherence_time * 1e3, color='g', 
                  linestyle='--', label=f'Tc = {autocorr_result.coherence_time*1e3:.3f} ms')
        
        ax.set_xlabel('Time Lag (ms)')
        ax.set_ylabel('|R(τ)|')
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        return ax
    
    @staticmethod
    def plot_frequency_response(freq: np.ndarray, magnitude: np.ndarray,
                               phase: np.ndarray = None, ax: plt.Axes = None,
                               title: str = "Frequency Response") -> plt.Axes:
        """
        Plot frequency response (magnitude and optionally phase).
        
        Args:
            freq: Frequency axis in Hz
            magnitude: |H(f)| or H(f) in dB
            phase: Phase of H(f) in radians (optional)
            ax: Matplotlib axes
            title: Plot title
            
        Returns:
            Matplotlib axes
        """
        if phase is not None:
            if ax is None:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            else:
                ax1, ax2 = ax
        else:
            if ax is None:
                fig, ax1 = plt.subplots(figsize=(10, 5))
            else:
                ax1 = ax  # Use provided ax directly
        
        # Convert frequency to MHz for readability
        freq_mhz = freq / 1e6
        
        # Magnitude plot
        ax1.plot(freq_mhz, magnitude, 'b-', linewidth=1.5)
        ax1.set_ylabel('Magnitude (dB)')
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        
        if phase is not None:
            ax2.plot(freq_mhz, np.unwrap(phase) * 180 / np.pi, 'r-', linewidth=1.5)
            ax2.set_xlabel('Frequency (MHz)')
            ax2.set_ylabel('Phase (degrees)')
            ax2.grid(True, alpha=0.3)
        else:
            ax1.set_xlabel('Frequency (MHz)')
        
        return ax1 if phase is None else (ax1, ax2)
    
    @staticmethod
    def plot_channel_magnitude(time: np.ndarray, magnitude: np.ndarray,
                              ax: plt.Axes = None,
                              title: str = "Channel Magnitude vs Time") -> plt.Axes:
        """
        Plot channel magnitude over time.
        
        Args:
            time: Time axis in seconds
            magnitude: Channel magnitude |h(t)|
            ax: Matplotlib axes
            title: Plot title
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        
        # Time in milliseconds
        time_ms = time * 1e3
        magnitude_db = 20 * np.log10(np.abs(magnitude) + 1e-10)
        
        ax.plot(time_ms, magnitude_db, 'b-', linewidth=0.8)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('|h(t)| (dB)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add mean line
        mean_db = np.mean(magnitude_db)
        ax.axhline(y=mean_db, color='r', linestyle='--', 
                  label=f'Mean: {mean_db:.2f} dB')
        ax.legend()
        
        return ax
    
    @staticmethod
    def plot_doppler_spectrum(freq: np.ndarray, spectrum: np.ndarray,
                             doppler_freq: float = None, ax: plt.Axes = None,
                             title: str = "Doppler Power Spectrum") -> plt.Axes:
        """
        Plot Doppler power spectrum.
        
        Args:
            freq: Frequency axis in Hz
            spectrum: Power spectrum
            doppler_freq: Maximum Doppler frequency (for reference lines)
            ax: Matplotlib axes
            title: Plot title
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        
        spectrum_db = 10 * np.log10(spectrum + 1e-10)
        
        ax.plot(freq, spectrum_db, 'b-', linewidth=1.5)
        
        if doppler_freq is not None:
            ax.axvline(x=doppler_freq, color='r', linestyle='--', 
                      label=f'+fd = {doppler_freq} Hz')
            ax.axvline(x=-doppler_freq, color='r', linestyle='--', 
                      label=f'-fd = {-doppler_freq} Hz')
            ax.set_xlim([-2*doppler_freq, 2*doppler_freq])
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power Spectral Density (dB)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax
    
    @staticmethod
    def plot_distribution_histogram(data: np.ndarray, 
                                   fit_result: FittingResult = None,
                                   ax: plt.Axes = None, bins: int = 50,
                                   title: str = "Distribution") -> plt.Axes:
        """
        Plot histogram with fitted distribution overlay.
        
        Args:
            data: Data to plot
            fit_result: FittingResult with fitted PDF
            ax: Matplotlib axes
            bins: Number of histogram bins
            title: Plot title
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        data = np.abs(data).flatten()
        
        # Histogram
        ax.hist(data, bins=bins, density=True, alpha=0.7, 
               color='steelblue', edgecolor='white', label='Observed')
        
        # Fitted PDF
        if fit_result is not None and fit_result.fitted_pdf is not None:
            ax.plot(fit_result.x_values, fit_result.fitted_pdf, 'r-', 
                   linewidth=2, label=f'Fitted {fit_result.distribution_name}')
            
            # Add fit statistics
            stats_text = (f"K-S stat: {fit_result.ks_statistic:.4f}\n"
                         f"p-value: {fit_result.ks_pvalue:.4f}")
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Magnitude')
        ax.set_ylabel('Probability Density')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    @staticmethod
    def plot_qq(theoretical: np.ndarray, observed: np.ndarray,
               distribution_name: str = "Theoretical", ax: plt.Axes = None,
               title: str = "Q-Q Plot") -> plt.Axes:
        """
        Plot Q-Q (Quantile-Quantile) plot.
        
        Args:
            theoretical: Theoretical quantiles
            observed: Observed quantiles
            distribution_name: Name of distribution for label
            ax: Matplotlib axes
            title: Plot title
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.scatter(theoretical, observed, alpha=0.5, s=20)
        
        # Add reference line
        min_val = min(np.min(theoretical), np.min(observed))
        max_val = max(np.max(theoretical), np.max(observed))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
               linewidth=2, label='Perfect fit')
        
        ax.set_xlabel(f'{distribution_name} Quantiles')
        ax.set_ylabel('Sample Quantiles')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        return ax
    
    @staticmethod
    def plot_correlation_matrix(corr_matrix: np.ndarray, ax: plt.Axes = None,
                               title: str = "Correlation Matrix") -> plt.Axes:
        """
        Plot correlation matrix as heatmap.
        
        Args:
            corr_matrix: Correlation matrix to plot
            ax: Matplotlib axes
            title: Plot title
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        im = ax.imshow(np.abs(corr_matrix), cmap='viridis', aspect='auto',
                      vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='|Correlation|')
        
        ax.set_xlabel('Element Index')
        ax.set_ylabel('Element Index')
        ax.set_title(title)
        
        return ax
    
    @staticmethod
    def create_analysis_dashboard(channel_data: np.ndarray, 
                                 sample_rate: float,
                                 doppler_freq: float = None) -> Figure:
        """
        Create comprehensive analysis dashboard with multiple plots.
        
        Args:
            channel_data: Channel impulse response data
            sample_rate: Sampling rate
            doppler_freq: Doppler frequency (optional)
            
        Returns:
            Matplotlib Figure with all analysis plots
        """
        fig = plt.figure(figsize=(16, 12))
        
        # If multipath (2D), flatten for some analyses
        if channel_data.ndim == 2:
            h_flat = channel_data[:, 0] if channel_data.shape[1] > 0 else channel_data.flatten()
        else:
            h_flat = channel_data.flatten()
        
        time = np.arange(len(h_flat)) / sample_rate
        
        # 1. Channel magnitude vs time
        ax1 = fig.add_subplot(2, 3, 1)
        ChannelPlotter.plot_channel_magnitude(time, h_flat, ax=ax1)
        
        # 2. PDP (if multipath)
        ax2 = fig.add_subplot(2, 3, 2)
        pdp = ChannelStatistics.compute_pdp(channel_data, sample_rate)
        ChannelPlotter.plot_pdp(pdp, ax=ax2)
        
        # 3. Autocorrelation
        ax3 = fig.add_subplot(2, 3, 3)
        autocorr = ChannelStatistics.compute_autocorrelation(h_flat, sample_rate)
        ChannelPlotter.plot_autocorrelation(autocorr, ax=ax3)
        
        # 4. Frequency response
        ax4 = fig.add_subplot(2, 3, 4)
        freq_resp = ChannelStatistics.compute_frequency_response(channel_data, sample_rate)
        ChannelPlotter.plot_frequency_response(
            freq_resp.frequencies, freq_resp.magnitude_db, ax=ax4
        )
        
        # 5. Doppler spectrum
        ax5 = fig.add_subplot(2, 3, 5)
        freq, spectrum = ChannelStatistics.compute_doppler_spectrum(h_flat, sample_rate)
        ChannelPlotter.plot_doppler_spectrum(freq, spectrum, doppler_freq, ax=ax5)
        
        # 6. Distribution histogram
        ax6 = fig.add_subplot(2, 3, 6)
        fit_result = DistributionFitter.fit_rayleigh(h_flat)
        ChannelPlotter.plot_distribution_histogram(h_flat, fit_result, ax=ax6,
                                                   title="Magnitude Distribution")
        
        fig.tight_layout()
        return fig
    
    @staticmethod
    def save_figure(fig: Figure, filepath: str, dpi: int = 150):
        """
        Save figure to file.
        
        Args:
            fig: Matplotlib figure
            filepath: Output file path
            dpi: Resolution
        """
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {filepath}")
