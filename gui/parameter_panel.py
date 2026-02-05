"""
Parameter Panel Widget.

Provides dynamic parameter configuration panels that change
based on the selected channel type.
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, Any, Callable, Optional


class ParameterPanel(ttk.LabelFrame):
    """
    Dynamic parameter configuration panel.
    
    Displays different parameters based on channel type selection.
    """
    
    def __init__(self, parent, on_change: Callable = None, **kwargs):
        """
        Initialize parameter panel.
        
        Args:
            parent: Parent widget
            on_change: Callback when parameters change
            **kwargs: Additional LabelFrame arguments
        """
        super().__init__(parent, text="参数配置", **kwargs)
        
        self.on_change = on_change
        self.param_widgets: Dict[str, tk.Widget] = {}
        self.param_vars: Dict[str, tk.Variable] = {}
        
        self._create_common_params()
        self._create_awgn_params()
        self._create_fading_params()
        self._create_multipath_params()
        
        # Default: show AWGN params
        self.set_channel_type('AWGN')
    
    def _create_param_row(self, parent, label: str, var_name: str, 
                         var_type: str = 'float', default: Any = 0,
                         row: int = 0, tooltip: str = None) -> ttk.Entry:
        """Create a labeled parameter entry row."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=5, pady=2)
        
        if var_type == 'float':
            var = tk.DoubleVar(value=default)
        elif var_type == 'int':
            var = tk.IntVar(value=default)
        else:
            var = tk.StringVar(value=str(default))
        
        self.param_vars[var_name] = var
        
        entry = ttk.Entry(parent, textvariable=var, width=15)
        entry.grid(row=row, column=1, sticky='ew', padx=5, pady=2)
        
        if tooltip:
            ttk.Label(parent, text=tooltip, font=('TkDefaultFont', 8)).grid(
                row=row, column=2, sticky='w', padx=5
            )
        
        self.param_widgets[var_name] = entry
        
        if self.on_change:
            var.trace_add('write', lambda *args: self.on_change())
        
        return entry
    
    def _create_common_params(self):
        """Create common parameters section."""
        self.common_frame = ttk.Frame(self)
        self.common_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(self.common_frame, text="通用参数", 
                 font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=0, columnspan=3, sticky='w')
        
        self._create_param_row(self.common_frame, "采样率 (Hz):", 'sample_rate', 
                              'float', 1e6, row=1, tooltip="采样频率")
        self._create_param_row(self.common_frame, "采样数:", 'num_samples', 
                              'int', 10000, row=2, tooltip="样本点数")
    
    def _create_awgn_params(self):
        """Create AWGN-specific parameters."""
        self.awgn_frame = ttk.Frame(self)
        
        ttk.Label(self.awgn_frame, text="AWGN 参数",
                 font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=0, columnspan=3, sticky='w')
        
        self._create_param_row(self.awgn_frame, "SNR (dB):", 'snr_db',
                              'float', 10.0, row=1, tooltip="信噪比")
    
    def _create_fading_params(self):
        """Create fading channel parameters (Rayleigh/Rice)."""
        self.fading_frame = ttk.Frame(self)
        
        ttk.Label(self.fading_frame, text="衰落参数",
                 font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=0, columnspan=3, sticky='w')
        
        self._create_param_row(self.fading_frame, "多普勒频移 (Hz):", 'doppler_freq',
                              'float', 100.0, row=1, tooltip="fd = v*fc/c")
        
        # Rice K-factor (only shown for Rice)
        self.k_factor_row = ttk.Frame(self.fading_frame)
        self.k_factor_row.grid(row=2, column=0, columnspan=3, sticky='ew')
        
        ttk.Label(self.k_factor_row, text="K 因子 (线性):").grid(row=0, column=0, sticky='w', padx=5)
        k_var = tk.DoubleVar(value=3.0)
        self.param_vars['k_factor'] = k_var
        k_entry = ttk.Entry(self.k_factor_row, textvariable=k_var, width=15)
        k_entry.grid(row=0, column=1, sticky='ew', padx=5)
        ttk.Label(self.k_factor_row, text="LOS/NLOS 功率比", font=('TkDefaultFont', 8)).grid(
            row=0, column=2, sticky='w', padx=5
        )
        self.param_widgets['k_factor'] = k_entry
    
    def _create_multipath_params(self):
        """Create multipath channel parameters."""
        self.multipath_frame = ttk.Frame(self)
        
        ttk.Label(self.multipath_frame, text="多径参数",
                 font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=0, columnspan=3, sticky='w')
        
        self._create_param_row(self.multipath_frame, "多径数:", 'num_paths',
                              'int', 4, row=1, tooltip="路径数量")
        self._create_param_row(self.multipath_frame, "RMS 延迟扩展 (s):", 'rms_delay_spread',
                              'float', 1e-6, row=2, tooltip="均方根延迟扩展")
        self._create_param_row(self.multipath_frame, "多普勒频移 (Hz):", 'multipath_doppler',
                              'float', 100.0, row=3, tooltip="各径多普勒")
        
        # PDP type selector
        ttk.Label(self.multipath_frame, text="功率分布类型:").grid(row=4, column=0, sticky='w', padx=5, pady=2)
        pdp_var = tk.StringVar(value='exponential')
        self.param_vars['pdp_type'] = pdp_var
        pdp_combo = ttk.Combobox(self.multipath_frame, textvariable=pdp_var, width=15,
                                values=['exponential', 'uniform', 'itu_ped_a', 'itu_veh_a'])
        pdp_combo.grid(row=4, column=1, sticky='ew', padx=5, pady=2)
        self.param_widgets['pdp_type'] = pdp_combo
    
    def set_channel_type(self, channel_type: str):
        """
        Update visible parameters based on channel type.
        
        Args:
            channel_type: One of 'AWGN', 'Rayleigh', 'Rice', 'Multipath'
        """
        # Hide all specific frames
        self.awgn_frame.pack_forget()
        self.fading_frame.pack_forget()
        self.multipath_frame.pack_forget()
        self.k_factor_row.grid_forget()
        
        if channel_type == 'AWGN':
            self.awgn_frame.pack(fill='x', padx=5, pady=5)
        elif channel_type == 'Rayleigh':
            self.fading_frame.pack(fill='x', padx=5, pady=5)
        elif channel_type == 'Rice':
            self.fading_frame.pack(fill='x', padx=5, pady=5)
            self.k_factor_row.grid(row=2, column=0, columnspan=3, sticky='ew')
        elif channel_type == 'Multipath':
            self.multipath_frame.pack(fill='x', padx=5, pady=5)
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current parameter values.
        
        Returns:
            Dictionary of parameter name -> value
        """
        params = {}
        for name, var in self.param_vars.items():
            try:
                params[name] = var.get()
            except tk.TclError:
                params[name] = 0
        return params
    
    def set_parameters(self, params: Dict[str, Any]):
        """
        Set parameter values.
        
        Args:
            params: Dictionary of parameter name -> value
        """
        for name, value in params.items():
            if name in self.param_vars:
                self.param_vars[name].set(value)


class ExportDialog(tk.Toplevel):
    """
    Export dialog for channel data.
    """
    
    def __init__(self, parent, callback: Callable):
        """
        Initialize export dialog.
        
        Args:
            parent: Parent window
            callback: Callback function(format, filepath)
        """
        super().__init__(parent)
        self.title("导出信道数据")
        self.geometry("400x250")
        self.resizable(False, False)
        
        self.callback = callback
        self.result = None
        
        self._create_widgets()
        
        # Make modal
        self.transient(parent)
        self.grab_set()
    
    def _create_widgets(self):
        """Create dialog widgets."""
        # Format selection
        format_frame = ttk.LabelFrame(self, text="导出格式")
        format_frame.pack(fill='x', padx=10, pady=10)
        
        self.format_var = tk.StringVar(value='npy')
        formats = [
            ('NumPy (.npy)', 'npy'),
            ('CSV (.csv)', 'csv'),
            ('MATLAB (.mat)', 'mat'),
            ('JSON (.json)', 'json'),
            ('完整导出包', 'package')
        ]
        
        for text, value in formats:
            ttk.Radiobutton(format_frame, text=text, value=value,
                           variable=self.format_var).pack(anchor='w', padx=10, pady=2)
        
        # Include options
        options_frame = ttk.LabelFrame(self, text="选项")
        options_frame.pack(fill='x', padx=10, pady=5)
        
        self.include_config = tk.BooleanVar(value=True)
        self.include_analysis = tk.BooleanVar(value=True)
        self.include_report = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(options_frame, text="包含配置信息", 
                       variable=self.include_config).pack(anchor='w', padx=10)
        ttk.Checkbutton(options_frame, text="包含分析结果",
                       variable=self.include_analysis).pack(anchor='w', padx=10)
        ttk.Checkbutton(options_frame, text="包含统计报告",
                       variable=self.include_report).pack(anchor='w', padx=10)
        
        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(btn_frame, text="导出", command=self._on_export).pack(side='right', padx=5)
        ttk.Button(btn_frame, text="取消", command=self.destroy).pack(side='right', padx=5)
    
    def _on_export(self):
        """Handle export button click."""
        from tkinter import filedialog
        
        fmt = self.format_var.get()
        
        if fmt == 'package':
            filepath = filedialog.askdirectory(title="选择导出目录")
        else:
            filetypes = {
                'npy': [('NumPy文件', '*.npy')],
                'csv': [('CSV文件', '*.csv')],
                'mat': [('MATLAB文件', '*.mat')],
                'json': [('JSON文件', '*.json')]
            }
            filepath = filedialog.asksaveasfilename(
                title="保存文件",
                filetypes=filetypes.get(fmt, [('所有文件', '*.*')]),
                defaultextension=f'.{fmt}'
            )
        
        if filepath:
            options = {
                'include_config': self.include_config.get(),
                'include_analysis': self.include_analysis.get(),
                'include_report': self.include_report.get()
            }
            self.callback(fmt, filepath, options)
            self.destroy()
