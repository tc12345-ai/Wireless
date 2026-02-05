"""
Multipath Time-Varying Channel Model.

Implements frequency-selective fading with configurable multipath
parameters including delays, path gains, and per-path Doppler.
"""

import numpy as np
from typing import List, Optional, Union
from .base import ChannelBase, ChannelResponse
from .rayleigh import RayleighChannel


class PowerDelayProfile:
    """Power Delay Profile models for multipath channels."""
    
    @staticmethod
    def exponential(num_paths: int, rms_delay_spread: float, 
                    sample_rate: float) -> tuple:
        """
        Generate exponential power delay profile.
        
        Args:
            num_paths: Number of multipath components
            rms_delay_spread: RMS delay spread in seconds
            sample_rate: Sampling rate
            
        Returns:
            Tuple of (delays, power_profile)
        """
        # Generate delays (uniformly spaced or exponentially spaced)
        max_delay = 5 * rms_delay_spread  # 5x RMS spread
        delays = np.linspace(0, max_delay, num_paths)
        
        # Exponential decay power profile
        # P(tau) = exp(-tau / sigma_tau)
        power_profile = np.exp(-delays / rms_delay_spread)
        
        # Normalize to unit total power
        power_profile = power_profile / np.sum(power_profile)
        
        return delays, power_profile
    
    @staticmethod
    def uniform(num_paths: int, max_delay: float) -> tuple:
        """
        Generate uniform power delay profile.
        
        Args:
            num_paths: Number of multipath components
            max_delay: Maximum delay in seconds
            
        Returns:
            Tuple of (delays, power_profile)
        """
        delays = np.linspace(0, max_delay, num_paths)
        power_profile = np.ones(num_paths) / num_paths
        
        return delays, power_profile
    
    @staticmethod
    def itu_pedestrian_a() -> tuple:
        """
        ITU Pedestrian A channel model.
        
        Returns:
            Tuple of (delays in seconds, power in dB)
        """
        delays = np.array([0, 110, 190, 410]) * 1e-9
        power_db = np.array([0, -9.7, -19.2, -22.8])
        power_linear = 10 ** (power_db / 10)
        power_linear = power_linear / np.sum(power_linear)
        
        return delays, power_linear
    
    @staticmethod
    def itu_vehicular_a() -> tuple:
        """
        ITU Vehicular A channel model.
        
        Returns:
            Tuple of (delays in seconds, power in dB)
        """
        delays = np.array([0, 310, 710, 1090, 1730, 2510]) * 1e-9
        power_db = np.array([0, -1.0, -9.0, -10.0, -15.0, -20.0])
        power_linear = 10 ** (power_db / 10)
        power_linear = power_linear / np.sum(power_linear)
        
        return delays, power_linear
    
    @staticmethod
    def custom(delays: np.ndarray, power_db: np.ndarray) -> tuple:
        """
        Create custom power delay profile.
        
        Args:
            delays: Array of path delays in seconds
            power_db: Array of path powers in dB
            
        Returns:
            Tuple of (delays, normalized linear power)
        """
        power_linear = 10 ** (power_db / 10)
        power_linear = power_linear / np.sum(power_linear)
        
        return delays, power_linear


class MultipathChannel(ChannelBase):
    """
    Multipath Time-Varying Fading Channel.
    
    Implements frequency-selective fading with multiple paths,
    each having independent Rayleigh fading with configurable Doppler.
    """
    
    def __init__(self, num_paths: int = 4, delays: np.ndarray = None,
                 path_gains: np.ndarray = None, doppler_freqs: Union[float, np.ndarray] = 100.0,
                 rms_delay_spread: float = 1e-6, sample_rate: float = 1e6,
                 num_samples: int = 1000, pdp_type: str = 'exponential'):
        """
        Initialize Multipath Channel.
        
        Args:
            num_paths: Number of multipath components
            delays: Path delays in seconds (optional, auto-generated if None)
            path_gains: Path gain powers (optional, auto-generated if None)
            doppler_freqs: Doppler frequency for each path (single value or array)
            rms_delay_spread: RMS delay spread in seconds (used if delays is None)
            sample_rate: Sampling rate in Hz
            num_samples: Number of time samples
            pdp_type: Power delay profile type ('exponential', 'uniform', 
                     'itu_ped_a', 'itu_veh_a', 'custom')
        """
        super().__init__(sample_rate, num_samples)
        self.num_paths = num_paths
        self.rms_delay_spread = rms_delay_spread
        self.pdp_type = pdp_type
        
        # Setup delays and path gains
        if delays is None or path_gains is None:
            self.delays, self.path_gains = self._generate_pdp()
        else:
            self.delays = np.asarray(delays)
            self.path_gains = np.asarray(path_gains)
            self.num_paths = len(self.delays)
        
        # Setup Doppler frequencies
        if isinstance(doppler_freqs, (int, float)):
            self.doppler_freqs = np.ones(self.num_paths) * doppler_freqs
        else:
            self.doppler_freqs = np.asarray(doppler_freqs)
        
        # Convert delays to samples
        self.delay_samples = np.round(self.delays * sample_rate).astype(int)
        
        # Create Rayleigh channel for each path
        self._path_channels = [
            RayleighChannel(
                doppler_freq=self.doppler_freqs[i],
                sample_rate=sample_rate,
                num_samples=num_samples
            )
            for i in range(self.num_paths)
        ]
    
    def _generate_pdp(self) -> tuple:
        """Generate power delay profile based on type."""
        if self.pdp_type == 'exponential':
            return PowerDelayProfile.exponential(
                self.num_paths, self.rms_delay_spread, self.sample_rate
            )
        elif self.pdp_type == 'uniform':
            max_delay = 5 * self.rms_delay_spread
            return PowerDelayProfile.uniform(self.num_paths, max_delay)
        elif self.pdp_type == 'itu_ped_a':
            delays, gains = PowerDelayProfile.itu_pedestrian_a()
            self.num_paths = len(delays)
            return delays, gains
        elif self.pdp_type == 'itu_veh_a':
            delays, gains = PowerDelayProfile.itu_vehicular_a()
            self.num_paths = len(delays)
            return delays, gains
        else:
            return PowerDelayProfile.exponential(
                self.num_paths, self.rms_delay_spread, self.sample_rate
            )
    
    def generate(self) -> ChannelResponse:
        """
        Generate time-varying multipath channel.
        
        Returns:
            ChannelResponse with shape (num_samples, max_delay_samples + 1)
        """
        time = np.arange(self.num_samples) / self.sample_rate
        max_delay_samples = np.max(self.delay_samples)
        
        # Channel impulse response: h[n, l] where n=time, l=delay tap
        h = np.zeros((self.num_samples, max_delay_samples + 1), dtype=complex)
        
        # Generate fading for each path
        path_fading = []
        for i, channel in enumerate(self._path_channels):
            channel.reset()
            response = channel.generate()
            fading = response.impulse_response
            
            # Scale by path gain
            fading = np.sqrt(self.path_gains[i]) * fading
            path_fading.append(fading)
            
            # Place in CIR at appropriate delay
            delay_idx = self.delay_samples[i]
            h[:, delay_idx] += fading
        
        self.channel_response = ChannelResponse(
            impulse_response=h,
            time_axis=time,
            delays=self.delays,
            path_gains=self.path_gains
        )
        
        return self.channel_response
    
    def apply(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply multipath channel to signal using convolution.
        
        Args:
            signal: Input signal
            
        Returns:
            Output signal after channel
        """
        if self.channel_response is None:
            self.generate()
        
        h = self.channel_response.impulse_response
        output = np.zeros(len(signal), dtype=complex)
        
        # Time-varying convolution
        for n in range(len(signal)):
            time_idx = min(n, h.shape[0] - 1)
            for l in range(h.shape[1]):
                if n - l >= 0:
                    output[n] += h[time_idx, l] * signal[n - l]
        
        return output
    
    def get_pdp(self) -> tuple:
        """
        Get power delay profile.
        
        Returns:
            Tuple of (delays, average_power)
        """
        if self.channel_response is None:
            self.generate()
        
        h = self.channel_response.impulse_response
        
        # Average power at each delay tap
        avg_power = np.mean(np.abs(h) ** 2, axis=0)
        delay_axis = np.arange(h.shape[1]) / self.sample_rate
        
        return delay_axis, avg_power
    
    def get_rms_delay_spread(self) -> float:
        """
        Calculate actual RMS delay spread from generated channel.
        
        Returns:
            RMS delay spread in seconds
        """
        delay_axis, avg_power = self.get_pdp()
        
        # Mean delay
        total_power = np.sum(avg_power)
        if total_power == 0:
            return 0.0
        
        mean_delay = np.sum(delay_axis * avg_power) / total_power
        
        # RMS delay spread
        rms_ds = np.sqrt(
            np.sum((delay_axis - mean_delay) ** 2 * avg_power) / total_power
        )
        
        return rms_ds
    
    def get_coherence_bandwidth(self) -> float:
        """
        Calculate coherence bandwidth (50% correlation).
        
        Returns:
            Coherence bandwidth in Hz
        """
        rms_ds = self.get_rms_delay_spread()
        if rms_ds == 0:
            return np.inf
        # B_c ≈ 1 / (5 * sigma_tau)
        return 1 / (5 * rms_ds)
    
    def __repr__(self) -> str:
        return (f"MultipathChannel(num_paths={self.num_paths}, "
                f"rms_delay_spread={self.rms_delay_spread:.2e}s, "
                f"doppler_freqs={self.doppler_freqs})")
