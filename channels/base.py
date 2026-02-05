"""
Base class for all channel models.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum


class ChannelType(Enum):
    """Enumeration of channel types."""
    AWGN = "AWGN"
    RAYLEIGH = "Rayleigh"
    RICE = "Rice"
    MULTIPATH = "Multipath"


@dataclass
class ChannelResponse:
    """Container for channel response data."""
    impulse_response: np.ndarray  # Channel impulse response h(t)
    frequency_response: Optional[np.ndarray] = None  # H(f)
    time_axis: Optional[np.ndarray] = None
    freq_axis: Optional[np.ndarray] = None
    delays: Optional[np.ndarray] = None  # Path delays for multipath
    path_gains: Optional[np.ndarray] = None  # Path gains for multipath


class ChannelBase(ABC):
    """Abstract base class for wireless channel models."""
    
    def __init__(self, sample_rate: float = 1e6, num_samples: int = 1000):
        """
        Initialize base channel.
        
        Args:
            sample_rate: Sampling rate in Hz
            num_samples: Number of samples to generate
        """
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.channel_response: Optional[ChannelResponse] = None
    
    @abstractmethod
    def generate(self) -> ChannelResponse:
        """
        Generate channel coefficients.
        
        Returns:
            ChannelResponse object containing channel data
        """
        pass
    
    def apply(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply channel to input signal.
        
        Args:
            signal: Input signal array
            
        Returns:
            Signal after passing through channel
        """
        if self.channel_response is None:
            self.generate()
        
        h = self.channel_response.impulse_response
        
        # Simple convolution for channel application
        if h.ndim == 1:
            # Time-invariant channel
            output = np.convolve(signal, h, mode='same')
        else:
            # Time-variant channel - apply element-wise multiplication
            # Assuming h has shape (num_samples, num_taps)
            output = np.zeros_like(signal, dtype=complex)
            for i in range(len(signal)):
                if i < h.shape[0]:
                    tap_len = min(h.shape[1], i + 1)
                    for k in range(tap_len):
                        if i - k >= 0:
                            output[i] += h[i, k] * signal[i - k]
        
        return output
    
    def get_frequency_response(self, nfft: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute frequency response of channel.
        
        Args:
            nfft: FFT size
            
        Returns:
            Tuple of (frequency axis, frequency response)
        """
        if self.channel_response is None:
            self.generate()
        
        h = self.channel_response.impulse_response
        
        # If time-variant, use the average or first snapshot
        if h.ndim > 1:
            h = h[0, :]
        
        H = np.fft.fft(h, nfft)
        freq = np.fft.fftfreq(nfft, 1 / self.sample_rate)
        
        # Shift to center
        H = np.fft.fftshift(H)
        freq = np.fft.fftshift(freq)
        
        return freq, H
    
    def reset(self):
        """Reset channel state."""
        self.channel_response = None
