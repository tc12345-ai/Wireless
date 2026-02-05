"""
Rayleigh Fading Channel Model.

Implements Rayleigh fading using the Jakes/Clarke model for realistic
time-varying fading simulation.
"""

import numpy as np
from .base import ChannelBase, ChannelResponse


class RayleighChannel(ChannelBase):
    """
    Rayleigh Fading Channel.
    
    Models flat fading when there is no line-of-sight component.
    Uses Jakes model to generate time-correlated fading coefficients.
    """
    
    def __init__(self, doppler_freq: float = 100.0, sample_rate: float = 1e6,
                 num_samples: int = 1000, num_sinusoids: int = 16):
        """
        Initialize Rayleigh Fading Channel.
        
        Args:
            doppler_freq: Maximum Doppler frequency in Hz (f_d = v*f_c/c)
            sample_rate: Sampling rate in Hz
            num_samples: Number of samples to generate
            num_sinusoids: Number of sinusoids for Jakes model
        """
        super().__init__(sample_rate, num_samples)
        self.doppler_freq = doppler_freq
        self.num_sinusoids = num_sinusoids
    
    def generate(self) -> ChannelResponse:
        """
        Generate Rayleigh fading coefficients using Jakes model.
        
        Returns:
            ChannelResponse with fading coefficients
        """
        N = self.num_sinusoids
        time = np.arange(self.num_samples) / self.sample_rate
        
        # Jakes model implementation
        h = np.zeros(self.num_samples, dtype=complex)
        
        for n in range(1, N + 1):
            # Angle of arrival for nth oscillator
            alpha_n = (2 * np.pi * n - np.pi + np.random.uniform(-np.pi, np.pi)) / (4 * N)
            beta_n = np.random.uniform(0, 2 * np.pi)
            
            # Frequency for nth oscillator
            f_n = self.doppler_freq * np.cos(alpha_n)
            
            # Add contribution
            h += np.exp(1j * (2 * np.pi * f_n * time + beta_n))
        
        # Normalize
        h = h / np.sqrt(N)
        
        # Alternative: Sum of sinusoids method (more accurate)
        h_real = np.zeros(self.num_samples)
        h_imag = np.zeros(self.num_samples)
        
        for n in range(N):
            theta_n = 2 * np.pi * (n + 1) / N
            phi_n = np.random.uniform(0, 2 * np.pi)
            psi_n = np.random.uniform(0, 2 * np.pi)
            
            f_d_n = self.doppler_freq * np.cos(theta_n)
            
            h_real += np.cos(2 * np.pi * f_d_n * time + phi_n)
            h_imag += np.sin(2 * np.pi * f_d_n * time + psi_n)
        
        h = (h_real + 1j * h_imag) / np.sqrt(N)
        
        self.channel_response = ChannelResponse(
            impulse_response=h,
            time_axis=time
        )
        
        return self.channel_response
    
    def generate_filtered(self) -> ChannelResponse:
        """
        Generate Rayleigh fading using filtered Gaussian noise method.
        
        This is an alternative method that filters white Gaussian noise
        with the Jakes Doppler spectrum.
        
        Returns:
            ChannelResponse with fading coefficients
        """
        time = np.arange(self.num_samples) / self.sample_rate
        
        # Generate white Gaussian noise
        wgn = (np.random.randn(self.num_samples) + 
               1j * np.random.randn(self.num_samples)) / np.sqrt(2)
        
        # Create Jakes/Clarke Doppler spectrum filter
        freq = np.fft.fftfreq(self.num_samples, 1 / self.sample_rate)
        
        # Jakes spectrum: S(f) = 1 / (pi * fd * sqrt(1 - (f/fd)^2))
        # with cutoff at |f| = fd
        doppler_filter = np.zeros(self.num_samples)
        valid_idx = np.abs(freq) < self.doppler_freq
        
        with np.errstate(divide='ignore', invalid='ignore'):
            doppler_filter[valid_idx] = 1 / np.sqrt(
                1 - (freq[valid_idx] / self.doppler_freq) ** 2
            )
        
        # Avoid infinity at edges
        doppler_filter = np.clip(doppler_filter, 0, 100)
        
        # Normalize filter
        doppler_filter = doppler_filter / np.sqrt(np.sum(doppler_filter ** 2))
        
        # Apply filter in frequency domain
        wgn_fft = np.fft.fft(wgn)
        h_fft = wgn_fft * doppler_filter
        h = np.fft.ifft(h_fft)
        
        # Normalize to unit power
        h = h / np.sqrt(np.mean(np.abs(h) ** 2))
        
        self.channel_response = ChannelResponse(
            impulse_response=h,
            time_axis=time
        )
        
        return self.channel_response
    
    def get_coherence_time(self) -> float:
        """
        Calculate channel coherence time.
        
        Returns:
            Coherence time in seconds
        """
        if self.doppler_freq == 0:
            return np.inf
        # T_c ≈ 9/(16*pi*fd) for 50% correlation
        return 9 / (16 * np.pi * self.doppler_freq)
    
    def get_doppler_spread(self) -> float:
        """
        Get Doppler spread (max Doppler frequency).
        
        Returns:
            Doppler spread in Hz
        """
        return self.doppler_freq
    
    def __repr__(self) -> str:
        return f"RayleighChannel(doppler_freq={self.doppler_freq}, sample_rate={self.sample_rate})"
