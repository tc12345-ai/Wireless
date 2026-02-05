"""
Channel Statistics Module.

Computes various statistical properties of wireless channels:
- Power Delay Profile (PDP)
- Autocorrelation function
- Frequency response
- Correlation matrix
- RMS delay spread
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class PDPResult:
    """Power Delay Profile result."""
    delays: np.ndarray  # Delay axis in seconds
    power: np.ndarray  # Power at each delay
    rms_delay_spread: float  # RMS delay spread
    mean_delay: float  # Mean excess delay
    max_delay: float  # Maximum excess delay


@dataclass
class AutocorrelationResult:
    """Autocorrelation result."""
    lags: np.ndarray  # Lag axis
    time_lags: np.ndarray  # Time lags in seconds
    autocorr: np.ndarray  # Autocorrelation values
    coherence_time: float  # 50% coherence time


@dataclass
class FrequencyResponseResult:
    """Frequency response result."""
    frequencies: np.ndarray  # Frequency axis in Hz
    magnitude: np.ndarray  # |H(f)|
    phase: np.ndarray  # angle(H(f))
    magnitude_db: np.ndarray  # 20*log10(|H(f)|)
    coherence_bandwidth: float  # Coherence bandwidth


class ChannelStatistics:
    """
    Compute statistical properties of channel responses.
    """
    
    @staticmethod
    def compute_pdp(impulse_response: np.ndarray, 
                    sample_rate: float) -> PDPResult:
        """
        Compute Power Delay Profile from channel impulse response.
        
        Args:
            impulse_response: Channel impulse response h[n] or h[n,l]
            sample_rate: Sampling rate in Hz
            
        Returns:
            PDPResult containing delay profile and statistics
        """
        h = np.asarray(impulse_response)
        
        # If time-varying (2D), average over time
        if h.ndim == 2:
            power = np.mean(np.abs(h) ** 2, axis=0)
        else:
            power = np.abs(h) ** 2
        
        # Create delay axis
        delays = np.arange(len(power)) / sample_rate
        
        # Compute statistics
        total_power = np.sum(power)
        
        if total_power > 0:
            # Mean delay
            mean_delay = np.sum(delays * power) / total_power
            
            # RMS delay spread
            rms_ds = np.sqrt(
                np.sum((delays - mean_delay) ** 2 * power) / total_power
            )
            
            # Maximum delay (at -10dB threshold)
            threshold = 0.1 * np.max(power)
            significant_idx = np.where(power > threshold)[0]
            max_delay = delays[significant_idx[-1]] if len(significant_idx) > 0 else 0
        else:
            mean_delay = 0
            rms_ds = 0
            max_delay = 0
        
        return PDPResult(
            delays=delays,
            power=power,
            rms_delay_spread=rms_ds,
            mean_delay=mean_delay,
            max_delay=max_delay
        )
    
    @staticmethod
    def compute_autocorrelation(channel_coeff: np.ndarray,
                                sample_rate: float,
                                max_lag: int = None) -> AutocorrelationResult:
        """
        Compute temporal autocorrelation of channel coefficients.
        
        Args:
            channel_coeff: Channel fading coefficients h[n]
            sample_rate: Sampling rate in Hz
            max_lag: Maximum lag to compute (default: N/2)
            
        Returns:
            AutocorrelationResult containing autocorrelation and coherence time
        """
        h = np.asarray(channel_coeff).flatten()
        N = len(h)
        
        if max_lag is None:
            max_lag = N // 2
        
        # Remove mean
        h_centered = h - np.mean(h)
        
        # Compute autocorrelation using FFT for efficiency
        h_fft = np.fft.fft(h_centered, n=2*N)
        autocorr_full = np.fft.ifft(h_fft * np.conj(h_fft)).real
        
        # Normalize
        autocorr = autocorr_full[:max_lag] / autocorr_full[0]
        
        # Lag axes
        lags = np.arange(max_lag)
        time_lags = lags / sample_rate
        
        # Find coherence time (50% correlation point)
        below_50 = np.where(np.abs(autocorr) < 0.5)[0]
        if len(below_50) > 0:
            coherence_time = time_lags[below_50[0]]
        else:
            coherence_time = time_lags[-1]
        
        return AutocorrelationResult(
            lags=lags,
            time_lags=time_lags,
            autocorr=autocorr,
            coherence_time=coherence_time
        )
    
    @staticmethod
    def compute_frequency_response(impulse_response: np.ndarray,
                                   sample_rate: float,
                                   nfft: int = 1024) -> FrequencyResponseResult:
        """
        Compute frequency response from impulse response.
        
        Args:
            impulse_response: Channel impulse response
            sample_rate: Sampling rate in Hz
            nfft: FFT size
            
        Returns:
            FrequencyResponseResult containing frequency response
        """
        h = np.asarray(impulse_response)
        
        # If time-varying, use first snapshot or average
        if h.ndim == 2:
            h = h[0, :]  # First time snapshot
        
        # Compute FFT
        H = np.fft.fft(h, nfft)
        freq = np.fft.fftfreq(nfft, 1 / sample_rate)
        
        # Shift to center
        H = np.fft.fftshift(H)
        freq = np.fft.fftshift(freq)
        
        # Compute magnitude and phase
        magnitude = np.abs(H)
        phase = np.angle(H)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        # Estimate coherence bandwidth from frequency correlation
        H_centered = H - np.mean(H)
        H_autocorr = np.abs(np.fft.ifft(np.abs(np.fft.fft(H_centered))**2))
        H_autocorr = H_autocorr / H_autocorr[0]
        
        freq_spacing = sample_rate / nfft
        below_50 = np.where(np.abs(H_autocorr[:nfft//2]) < 0.5)[0]
        if len(below_50) > 0:
            coherence_bw = below_50[0] * freq_spacing
        else:
            coherence_bw = sample_rate / 2
        
        return FrequencyResponseResult(
            frequencies=freq,
            magnitude=magnitude,
            phase=phase,
            magnitude_db=magnitude_db,
            coherence_bandwidth=coherence_bw
        )
    
    @staticmethod
    def compute_correlation_matrix(channel_samples: np.ndarray) -> np.ndarray:
        """
        Compute correlation matrix of channel samples.
        
        Args:
            channel_samples: Array of channel samples (N x M)
                            N = number of snapshots, M = number of elements
            
        Returns:
            Correlation matrix (M x M)
        """
        samples = np.asarray(channel_samples)
        
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        
        # Compute covariance matrix
        samples_centered = samples - np.mean(samples, axis=0)
        cov_matrix = np.dot(samples_centered.conj().T, samples_centered) / (samples.shape[0] - 1)
        
        # Normalize to correlation matrix
        std_diag = np.sqrt(np.diag(cov_matrix))
        std_outer = np.outer(std_diag, std_diag)
        corr_matrix = cov_matrix / (std_outer + 1e-10)
        
        return corr_matrix
    
    @staticmethod
    def compute_doppler_spectrum(channel_coeff: np.ndarray,
                                 sample_rate: float,
                                 nfft: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Doppler power spectrum of channel coefficients.
        
        Args:
            channel_coeff: Channel fading coefficients
            sample_rate: Sampling rate in Hz
            nfft: FFT size
            
        Returns:
            Tuple of (frequency axis, Doppler spectrum)
        """
        h = np.asarray(channel_coeff).flatten()
        
        # Compute power spectrum
        h_fft = np.fft.fft(h, nfft)
        spectrum = np.abs(h_fft) ** 2 / len(h)
        
        freq = np.fft.fftfreq(nfft, 1 / sample_rate)
        
        # Shift to center
        spectrum = np.fft.fftshift(spectrum)
        freq = np.fft.fftshift(freq)
        
        return freq, spectrum
    
    @staticmethod
    def compute_level_crossing_rate(magnitude: np.ndarray,
                                   threshold: float,
                                   sample_rate: float) -> float:
        """
        Compute level crossing rate of channel magnitude.
        
        Args:
            magnitude: Channel magnitude |h[n]|
            threshold: Threshold level
            sample_rate: Sampling rate in Hz
            
        Returns:
            Level crossing rate (crossings per second)
        """
        mag = np.asarray(magnitude).flatten()
        
        # Find upward crossings
        below = mag[:-1] < threshold
        above = mag[1:] >= threshold
        crossings = np.sum(below & above)
        
        duration = len(mag) / sample_rate
        lcr = crossings / duration
        
        return lcr
    
    @staticmethod
    def compute_average_fade_duration(magnitude: np.ndarray,
                                      threshold: float,
                                      sample_rate: float) -> float:
        """
        Compute average fade duration below threshold.
        
        Args:
            magnitude: Channel magnitude |h[n]|
            threshold: Threshold level
            sample_rate: Sampling rate in Hz
            
        Returns:
            Average fade duration in seconds
        """
        mag = np.asarray(magnitude).flatten()
        
        # Find regions below threshold
        below = mag < threshold
        
        # Count samples below threshold
        samples_below = np.sum(below)
        
        # Count fade events (transitions from above to below)
        fade_starts = np.sum((~below[:-1]) & below[1:])
        
        if fade_starts == 0:
            return 0.0
        
        avg_duration = (samples_below / fade_starts) / sample_rate
        
        return avg_duration
