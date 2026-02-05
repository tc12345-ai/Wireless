"""
Channel Configuration Data Classes.

Provides dataclass-based configuration for all channel types
with validation and serialization support.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Union
from enum import Enum
import numpy as np
import json


class ChannelType(Enum):
    """Enumeration of available channel types."""
    AWGN = "AWGN"
    RAYLEIGH = "Rayleigh"
    RICE = "Rice"
    MULTIPATH = "Multipath"


class PDPType(Enum):
    """Power Delay Profile types."""
    EXPONENTIAL = "exponential"
    UNIFORM = "uniform"
    ITU_PED_A = "itu_ped_a"
    ITU_VEH_A = "itu_veh_a"
    CUSTOM = "custom"


@dataclass
class ChannelConfig:
    """
    Comprehensive channel configuration.
    
    Contains all parameters needed to configure any channel type.
    Unused parameters for a specific channel type are ignored.
    """
    # Common parameters
    channel_type: ChannelType = ChannelType.AWGN
    sample_rate: float = 1e6  # Hz
    num_samples: int = 1000
    
    # AWGN parameters
    snr_db: float = 10.0
    
    # Fading parameters (Rayleigh/Rice)
    doppler_freq: float = 100.0  # Hz
    k_factor: float = 3.0  # Rice K-factor (linear)
    los_doppler_shift: Optional[float] = None  # Hz
    
    # Multipath parameters
    num_paths: int = 4
    rms_delay_spread: float = 1e-6  # seconds
    pdp_type: PDPType = PDPType.EXPONENTIAL
    custom_delays: Optional[List[float]] = None  # seconds
    custom_powers_db: Optional[List[float]] = None  # dB
    path_doppler_freqs: Optional[List[float]] = None  # Per-path Doppler
    
    def validate(self) -> List[str]:
        """
        Validate configuration parameters.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if self.sample_rate <= 0:
            errors.append("Sample rate must be positive")
        
        if self.num_samples <= 0:
            errors.append("Number of samples must be positive")
        
        if self.doppler_freq < 0:
            errors.append("Doppler frequency cannot be negative")
        
        if self.k_factor < 0:
            errors.append("K-factor cannot be negative")
        
        if self.num_paths <= 0:
            errors.append("Number of paths must be positive")
        
        if self.rms_delay_spread < 0:
            errors.append("RMS delay spread cannot be negative")
        
        if self.pdp_type == PDPType.CUSTOM:
            if self.custom_delays is None or self.custom_powers_db is None:
                errors.append("Custom PDP requires delays and powers to be specified")
            elif len(self.custom_delays) != len(self.custom_powers_db):
                errors.append("Custom delays and powers must have same length")
        
        return errors
    
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, np.ndarray):
                result[key] = value.tolist()
            else:
                result[key] = value
        return result
    
    def to_json(self) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ChannelConfig':
        """Create configuration from dictionary."""
        # Convert string enums back to Enum types
        if 'channel_type' in data and isinstance(data['channel_type'], str):
            data['channel_type'] = ChannelType(data['channel_type'])
        if 'pdp_type' in data and isinstance(data['pdp_type'], str):
            data['pdp_type'] = PDPType(data['pdp_type'])
        
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ChannelConfig':
        """Create configuration from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_doppler_spread_hz(self) -> float:
        """Get maximum Doppler spread in Hz."""
        return self.doppler_freq
    
    def get_coherence_time(self) -> float:
        """Calculate coherence time in seconds."""
        if self.doppler_freq == 0:
            return np.inf
        return 9 / (16 * np.pi * self.doppler_freq)
    
    def get_coherence_bandwidth(self) -> float:
        """Calculate coherence bandwidth in Hz."""
        if self.rms_delay_spread == 0:
            return np.inf
        return 1 / (5 * self.rms_delay_spread)
    
    def __repr__(self) -> str:
        return (f"ChannelConfig(type={self.channel_type.value}, "
                f"fs={self.sample_rate:.0e}, N={self.num_samples})")


# Preset configurations
PRESET_CONFIGS = {
    'awgn_10db': ChannelConfig(
        channel_type=ChannelType.AWGN,
        snr_db=10.0
    ),
    'rayleigh_slow': ChannelConfig(
        channel_type=ChannelType.RAYLEIGH,
        doppler_freq=10.0,
        sample_rate=1e6,
        num_samples=10000
    ),
    'rayleigh_fast': ChannelConfig(
        channel_type=ChannelType.RAYLEIGH,
        doppler_freq=200.0,
        sample_rate=1e6,
        num_samples=10000
    ),
    'rice_los_dominant': ChannelConfig(
        channel_type=ChannelType.RICE,
        k_factor=10.0,
        doppler_freq=100.0
    ),
    'multipath_urban': ChannelConfig(
        channel_type=ChannelType.MULTIPATH,
        num_paths=6,
        rms_delay_spread=1e-6,
        pdp_type=PDPType.EXPONENTIAL,
        doppler_freq=50.0
    ),
    'itu_pedestrian': ChannelConfig(
        channel_type=ChannelType.MULTIPATH,
        pdp_type=PDPType.ITU_PED_A,
        doppler_freq=5.0
    ),
    'itu_vehicular': ChannelConfig(
        channel_type=ChannelType.MULTIPATH,
        pdp_type=PDPType.ITU_VEH_A,
        doppler_freq=100.0
    )
}
