#!/usr/bin/env python3
"""
Wireless Channel Simulator
无线信道建模与参数化仿真软件

Main entry point for the application.

Features:
- AWGN, Rayleigh, Rice, Multipath channel generation
- Configurable parameters (paths, PSD, delay spread, Doppler)
- Output: PDP, autocorrelation, frequency response, correlation matrix
- GUI for parameter configuration
- Automatic channel sample export
- Statistical property verification (distribution fitting)

Usage:
    python main.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.main_window import run_app


def main():
    """Main entry point."""
    print("=" * 60)
    print("无线信道建模与参数化仿真软件")
    print("Wireless Channel Modeling and Simulation Software")
    print("=" * 60)
    print("\n启动图形界面...\n")
    
    run_app()


if __name__ == '__main__':
    main()
