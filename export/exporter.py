"""
Channel Data Exporter Module.

Exports channel samples and analysis results to various formats:
- NumPy (.npy, .npz)
- CSV
- MATLAB (.mat)
- JSON
- Analysis report (text/HTML)
"""

import numpy as np
import json
import csv
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


class ChannelExporter:
    """
    Export channel data and analysis results to various formats.
    """
    
    SUPPORTED_FORMATS = ['npy', 'npz', 'csv', 'mat', 'json', 'txt']
    
    @staticmethod
    def export_to_numpy(data: np.ndarray, filepath: str) -> str:
        """
        Export channel data to NumPy .npy format.
        
        Args:
            data: Channel data array
            filepath: Output file path
            
        Returns:
            Saved file path
        """
        filepath = Path(filepath)
        if filepath.suffix != '.npy':
            filepath = filepath.with_suffix('.npy')
        
        np.save(filepath, data)
        print(f"Data saved to {filepath}")
        return str(filepath)
    
    @staticmethod
    def export_to_npz(data_dict: Dict[str, np.ndarray], filepath: str,
                     compressed: bool = True) -> str:
        """
        Export multiple arrays to NumPy .npz format.
        
        Args:
            data_dict: Dictionary of name -> array
            filepath: Output file path
            compressed: Whether to compress the archive
            
        Returns:
            Saved file path
        """
        filepath = Path(filepath)
        if filepath.suffix != '.npz':
            filepath = filepath.with_suffix('.npz')
        
        if compressed:
            np.savez_compressed(filepath, **data_dict)
        else:
            np.savez(filepath, **data_dict)
        
        print(f"Data saved to {filepath}")
        return str(filepath)
    
    @staticmethod
    def export_to_csv(data: np.ndarray, filepath: str, 
                     headers: List[str] = None,
                     include_time: bool = True,
                     sample_rate: float = 1e6) -> str:
        """
        Export channel data to CSV format.
        
        Args:
            data: Channel data (1D or 2D array)
            filepath: Output file path
            headers: Column headers
            include_time: Include time column
            sample_rate: Sampling rate for time column
            
        Returns:
            Saved file path
        """
        filepath = Path(filepath)
        if filepath.suffix != '.csv':
            filepath = filepath.with_suffix('.csv')
        
        data = np.asarray(data)
        
        # Flatten if 1D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Prepare time column
        num_samples = data.shape[0]
        time = np.arange(num_samples) / sample_rate
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write headers
            if headers is None:
                if np.iscomplexobj(data):
                    headers = [f'h{i}_real,h{i}_imag' for i in range(data.shape[1])]
                else:
                    headers = [f'h{i}' for i in range(data.shape[1])]
            
            if include_time:
                header_row = ['time(s)'] + headers
            else:
                header_row = headers
            writer.writerow(header_row)
            
            # Write data
            for i in range(num_samples):
                if include_time:
                    row = [time[i]]
                else:
                    row = []
                
                for j in range(data.shape[1]):
                    if np.iscomplexobj(data):
                        row.extend([data[i, j].real, data[i, j].imag])
                    else:
                        row.append(data[i, j])
                
                writer.writerow(row)
        
        print(f"Data saved to {filepath}")
        return str(filepath)
    
    @staticmethod
    def export_to_mat(data_dict: Dict[str, Any], filepath: str) -> str:
        """
        Export data to MATLAB .mat format.
        
        Args:
            data_dict: Dictionary of variable name -> data
            filepath: Output file path
            
        Returns:
            Saved file path
        """
        try:
            from scipy.io import savemat
        except ImportError:
            raise ImportError("scipy is required for .mat export")
        
        filepath = Path(filepath)
        if filepath.suffix != '.mat':
            filepath = filepath.with_suffix('.mat')
        
        # Convert complex arrays to MATLAB-compatible format
        mat_dict = {}
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                mat_dict[key] = value
            elif isinstance(value, (int, float)):
                mat_dict[key] = value
            elif isinstance(value, str):
                mat_dict[key] = value
            elif isinstance(value, dict):
                # Flatten nested dicts
                for k, v in value.items():
                    mat_dict[f'{key}_{k}'] = v
        
        savemat(filepath, mat_dict)
        print(f"Data saved to {filepath}")
        return str(filepath)
    
    @staticmethod
    def export_to_json(config_dict: Dict[str, Any], filepath: str) -> str:
        """
        Export configuration/metadata to JSON format.
        
        Args:
            config_dict: Configuration dictionary
            filepath: Output file path
            
        Returns:
            Saved file path
        """
        filepath = Path(filepath)
        if filepath.suffix != '.json':
            filepath = filepath.with_suffix('.json')
        
        # Custom encoder for numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    if np.iscomplexobj(obj):
                        return {
                            'real': obj.real.tolist(),
                            'imag': obj.imag.tolist()
                        }
                    return obj.tolist()
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, complex):
                    return {'real': obj.real, 'imag': obj.imag}
                return super().default(obj)
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, cls=NumpyEncoder)
        
        print(f"Data saved to {filepath}")
        return str(filepath)
    
    @staticmethod
    def export_report(report_text: str, filepath: str, 
                     format: str = 'txt') -> str:
        """
        Export analysis report to text or HTML format.
        
        Args:
            report_text: Report content
            filepath: Output file path
            format: 'txt' or 'html'
            
        Returns:
            Saved file path
        """
        filepath = Path(filepath)
        
        if format == 'html':
            if filepath.suffix != '.html':
                filepath = filepath.with_suffix('.html')
            
            # Convert to HTML
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Channel Analysis Report</title>
    <style>
        body {{ font-family: 'Courier New', monospace; padding: 20px; }}
        pre {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
        h1 {{ color: #333; }}
    </style>
</head>
<body>
    <h1>Channel Analysis Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <pre>{report_text}</pre>
</body>
</html>"""
            with open(filepath, 'w') as f:
                f.write(html_content)
        else:
            if filepath.suffix != '.txt':
                filepath = filepath.with_suffix('.txt')
            
            with open(filepath, 'w') as f:
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(report_text)
        
        print(f"Report saved to {filepath}")
        return str(filepath)
    
    @staticmethod
    def export_batch(data: np.ndarray, config: Dict[str, Any],
                    output_dir: str, base_name: str = 'channel',
                    formats: List[str] = None) -> Dict[str, str]:
        """
        Export channel data to multiple formats at once.
        
        Args:
            data: Channel data
            config: Configuration dictionary
            output_dir: Output directory
            base_name: Base filename (without extension)
            formats: List of formats to export ['npy', 'csv', 'mat', 'json']
            
        Returns:
            Dictionary of format -> saved filepath
        """
        if formats is None:
            formats = ['npy', 'csv', 'json']
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        for fmt in formats:
            filepath = output_dir / f"{base_name}.{fmt}"
            
            if fmt == 'npy':
                saved_files[fmt] = ChannelExporter.export_to_numpy(data, str(filepath))
            elif fmt == 'csv':
                sample_rate = config.get('sample_rate', 1e6)
                saved_files[fmt] = ChannelExporter.export_to_csv(
                    data, str(filepath), sample_rate=sample_rate
                )
            elif fmt == 'mat':
                mat_data = {
                    'channel_data': data,
                    'sample_rate': config.get('sample_rate', 1e6),
                    'num_samples': len(data),
                    'channel_type': str(config.get('channel_type', 'unknown'))
                }
                saved_files[fmt] = ChannelExporter.export_to_mat(mat_data, str(filepath))
            elif fmt == 'json':
                saved_files[fmt] = ChannelExporter.export_to_json(config, str(filepath))
        
        return saved_files
    
    @staticmethod
    def create_export_package(data: np.ndarray, 
                             config: Dict[str, Any],
                             analysis_results: Dict[str, Any],
                             report: str,
                             output_dir: str,
                             name: str = None) -> str:
        """
        Create complete export package with all data and results.
        
        Args:
            data: Channel data
            config: Configuration dictionary
            analysis_results: Dictionary of analysis results
            report: Text report
            output_dir: Output directory
            name: Package name (auto-generated if None)
            
        Returns:
            Path to export directory
        """
        if name is None:
            name = f"channel_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        output_dir = Path(output_dir) / name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export channel data
        ChannelExporter.export_to_numpy(data, str(output_dir / 'channel_data.npy'))
        ChannelExporter.export_to_csv(data, str(output_dir / 'channel_data.csv'),
                                     sample_rate=config.get('sample_rate', 1e6))
        
        # Export MATLAB format
        try:
            mat_data = {
                'h': data,
                'fs': config.get('sample_rate', 1e6),
                **{k: v for k, v in analysis_results.items() if isinstance(v, (np.ndarray, float, int))}
            }
            ChannelExporter.export_to_mat(mat_data, str(output_dir / 'channel_data.mat'))
        except ImportError:
            print("scipy not available, skipping .mat export")
        
        # Export configuration
        ChannelExporter.export_to_json(config, str(output_dir / 'config.json'))
        
        # Export analysis results
        ChannelExporter.export_to_json(analysis_results, str(output_dir / 'analysis.json'))
        
        # Export report
        ChannelExporter.export_report(report, str(output_dir / 'report.txt'))
        ChannelExporter.export_report(report, str(output_dir / 'report.html'), format='html')
        
        print(f"\nExport package created at: {output_dir}")
        return str(output_dir)
