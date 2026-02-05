"""
Rice (Rician) Fading Channel Model.

Implements Rician fading which includes a line-of-sight (LOS) component
in addition to scattered (NLOS) components.
"""

import numpy as np
from .base import ChannelBase, ChannelResponse
from .rayleigh import RayleighChannel


class RiceChannel(ChannelBase):
    """
    Rice (Rician) Fading Channel.
    
    Models fading when there is a dominant line-of-sight component.
    The K-factor determines the ratio of LOS to NLOS power.
    """
    
    def __init__(self, k_factor: float = 3.0, doppler_freq: float = 100.0,
                 sample_rate: float = 1e6, num_samples: int = 1000,
                 los_doppler_shift: float = None):
        """
        Initialize Rician Fading Channel.
        
        Args:
            k_factor: Rice K-factor (LOS power / NLOS power) in linear scale
            doppler_freq: Maximum Doppler frequency for scattered component (Hz)
            sample_rate: Sampling rate in Hz
            num_samples: Number of samples to generate
            los_doppler_shift: Doppler shift of LOS component (Hz). 
                              If None, defaults to doppler_freq
        """
        super().__init__(sample_rate, num_samples)
        self.k_factor = k_factor
        self.doppler_freq = doppler_freq
        self.los_doppler_shift = los_doppler_shift if los_doppler_shift is not None else doppler_freq
        
        # Create Rayleigh channel for scattered component
        self._rayleigh = RayleighChannel(
            doppler_freq=doppler_freq,
            sample_rate=sample_rate,
            num_samples=num_samples
        )
    
    @property
    def k_factor_db(self) -> float:
        """Get K-factor in dB."""
        return 10 * np.log10(self.k_factor)
    
    @k_factor_db.setter
    def k_factor_db(self, value: float):
        """Set K-factor from dB value."""
        self.k_factor = 10 ** (value / 10)
    
    def generate(self) -> ChannelResponse:
        """
        Generate Rician fading coefficients.
        
        h = sqrt(K/(K+1)) * h_LOS + sqrt(1/(K+1)) * h_NLOS
        
        Returns:
            ChannelResponse with fading coefficients
        """
        time = np.arange(self.num_samples) / self.sample_rate
        
        # Calculate component powers
        los_amplitude = np.sqrt(self.k_factor / (self.k_factor + 1))
        nlos_amplitude = np.sqrt(1 / (self.k_factor + 1))
        
        # LOS component: deterministic sinusoid with Doppler shift
        los_phase = np.random.uniform(0, 2 * np.pi)  # Random initial phase
        h_los = los_amplitude * np.exp(1j * (2 * np.pi * self.los_doppler_shift * time + los_phase))
        
        # NLOS component: Rayleigh fading
        self._rayleigh.reset()
        rayleigh_response = self._rayleigh.generate()
        h_nlos = nlos_amplitude * rayleigh_response.impulse_response
        
        # Combine components
        h = h_los + h_nlos
        
        self.channel_response = ChannelResponse(
            impulse_response=h,
            time_axis=time
        )
        
        return self.channel_response
    
    def get_los_power_ratio(self) -> float:
        """
        Get the ratio of LOS power to total power.
        
        Returns:
            LOS power ratio (0 to 1)
        """
        return self.k_factor / (self.k_factor + 1)
    
    def get_nlos_power_ratio(self) -> float:
        """
        Get the ratio of NLOS power to total power.
        
        Returns:
            NLOS power ratio (0 to 1)
        """
        return 1 / (self.k_factor + 1)
    
    @staticmethod
    def from_k_factor_db(k_factor_db: float, **kwargs) -> 'RiceChannel':
        """
        Create RiceChannel from K-factor in dB.
        
        Args:
            k_factor_db: K-factor in dB
            **kwargs: Other arguments to pass to constructor
            
        Returns:
            RiceChannel instance
        """
        k_factor = 10 ** (k_factor_db / 10)
        return RiceChannel(k_factor=k_factor, **kwargs)
    
    def __repr__(self) -> str:
        return (f"RiceChannel(k_factor={self.k_factor:.2f} ({self.k_factor_db:.1f} dB), "
                f"doppler_freq={self.doppler_freq})")
