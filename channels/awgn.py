"""
AWGN (Additive White Gaussian Noise) Channel Model.
"""

import numpy as np
from .base import ChannelBase, ChannelResponse


class AWGNChannel(ChannelBase):
    """
    Additive White Gaussian Noise Channel.
    
    The simplest channel model - adds Gaussian noise to the signal.
    """
    
    def __init__(self, snr_db: float = 10.0, sample_rate: float = 1e6, 
                 num_samples: int = 1000):
        """
        Initialize AWGN Channel.
        
        Args:
            snr_db: Signal-to-Noise Ratio in dB
            sample_rate: Sampling rate in Hz
            num_samples: Number of samples to generate
        """
        super().__init__(sample_rate, num_samples)
        self.snr_db = snr_db
    
    @property
    def noise_power(self) -> float:
        """Calculate noise power from SNR (assuming unit signal power)."""
        return 10 ** (-self.snr_db / 10)
    
    @property
    def noise_std(self) -> float:
        """Calculate noise standard deviation."""
        return np.sqrt(self.noise_power / 2)  # Divide by 2 for complex noise
    
    def generate(self) -> ChannelResponse:
        """
        Generate AWGN samples.
        
        Returns:
            ChannelResponse with noise samples as impulse response
        """
        # Generate complex Gaussian noise
        noise_real = np.random.normal(0, self.noise_std, self.num_samples)
        noise_imag = np.random.normal(0, self.noise_std, self.num_samples)
        noise = noise_real + 1j * noise_imag
        
        # Time axis
        time_axis = np.arange(self.num_samples) / self.sample_rate
        
        self.channel_response = ChannelResponse(
            impulse_response=noise,
            time_axis=time_axis
        )
        
        return self.channel_response
    
    def apply(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply AWGN to signal.
        
        Args:
            signal: Input signal
            
        Returns:
            Signal with added noise
        """
        # Calculate actual noise power based on signal power
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = signal_power * (10 ** (-self.snr_db / 10))
        noise_std = np.sqrt(noise_power / 2)
        
        # Generate noise
        noise = np.random.normal(0, noise_std, signal.shape) + \
                1j * np.random.normal(0, noise_std, signal.shape)
        
        return signal + noise
    
    def generate_noise(self, shape: tuple) -> np.ndarray:
        """
        Generate noise samples with given shape.
        
        Args:
            shape: Shape of noise array to generate
            
        Returns:
            Complex noise array
        """
        noise = np.random.normal(0, self.noise_std, shape) + \
                1j * np.random.normal(0, self.noise_std, shape)
        return noise
    
    def __repr__(self) -> str:
        return f"AWGNChannel(snr_db={self.snr_db}, sample_rate={self.sample_rate})"
