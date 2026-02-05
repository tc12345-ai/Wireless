"""
Main Application Window.

Provides the main GUI for the channel simulator with:
- Channel type selection
- Parameter configuration
- Real-time visualization
- Export functionality
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from channels import AWGNChannel, RayleighChannel, RiceChannel, MultipathChannel
from config import ChannelConfig, ChannelType
from analysis import ChannelStatistics, DistributionFitter
from visualization import ChannelPlotter
from export import ChannelExporter
from gui.parameter_panel import ParameterPanel, ExportDialog


class MainWindow:
    """
    Main application window for channel simulator.
    """
    
    def __init__(self, root: tk.Tk):
        """
        Initialize main window.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("无线信道建模与仿真软件 - Wireless Channel Simulator")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # State variables
        self.current_channel = None
        self.channel_data = None
        self.analysis_results = {}
        
        # Setup UI
        self._setup_styles()
        self._create_menu()
        self._create_main_layout()
        self._create_status_bar()
        
        # Initial state
        self._on_channel_type_changed()
    
    def _setup_styles(self):
        """Configure ttk styles."""
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Microsoft YaHei', 14, 'bold'))
        style.configure('Header.TLabel', font=('Microsoft YaHei', 11, 'bold'))
        style.configure('Status.TLabel', font=('Microsoft YaHei', 9))
    
    def _create_menu(self):
        """Create application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="导出数据...", command=self._show_export_dialog)
        file_menu.add_command(label="保存配置...", command=self._save_config)
        file_menu.add_command(label="载入配置...", command=self._load_config)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="视图", menu=view_menu)
        view_menu.add_command(label="刷新图表", command=self._refresh_plots)
        view_menu.add_command(label="重置视图", command=self._reset_view)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="关于", command=self._show_about)
    
    def _create_main_layout(self):
        """Create main window layout."""
        # Main paned window
        self.main_paned = ttk.PanedWindow(self.root, orient='horizontal')
        self.main_paned.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Left panel - Controls
        self.left_panel = self._create_left_panel()
        self.main_paned.add(self.left_panel, weight=0)
        
        # Right panel - Visualization
        self.right_panel = self._create_right_panel()
        self.main_paned.add(self.right_panel, weight=1)
    
    def _create_left_panel(self) -> ttk.Frame:
        """Create left control panel."""
        panel = ttk.Frame(self.main_paned)
        
        # Title
        ttk.Label(panel, text="信道配置", style='Title.TLabel').pack(pady=10)
        
        # Channel type selection
        type_frame = ttk.LabelFrame(panel, text="信道类型")
        type_frame.pack(fill='x', padx=5, pady=5)
        
        self.channel_type_var = tk.StringVar(value='AWGN')
        channel_types = [
            ('AWGN (加性白噪声)', 'AWGN'),
            ('Rayleigh (瑞利衰落)', 'Rayleigh'),
            ('Rice (莱斯衰落)', 'Rice'),
            ('Multipath (多径信道)', 'Multipath')
        ]
        
        for text, value in channel_types:
            rb = ttk.Radiobutton(type_frame, text=text, value=value,
                                variable=self.channel_type_var,
                                command=self._on_channel_type_changed)
            rb.pack(anchor='w', padx=10, pady=2)
        
        # Parameter panel
        self.param_panel = ParameterPanel(panel, on_change=self._on_params_changed)
        self.param_panel.pack(fill='x', padx=5, pady=5)
        
        # Action buttons
        btn_frame = ttk.Frame(panel)
        btn_frame.pack(fill='x', padx=5, pady=10)
        
        ttk.Button(btn_frame, text="生成信道", 
                  command=self._generate_channel).pack(fill='x', pady=2)
        ttk.Button(btn_frame, text="统计分析",
                  command=self._run_analysis).pack(fill='x', pady=2)
        ttk.Button(btn_frame, text="分布检验",
                  command=self._run_distribution_test).pack(fill='x', pady=2)
        ttk.Button(btn_frame, text="导出数据",
                  command=self._show_export_dialog).pack(fill='x', pady=2)
        
        # Quick info panel
        self.info_frame = ttk.LabelFrame(panel, text="信道信息")
        self.info_frame.pack(fill='x', padx=5, pady=5)
        
        self.info_text = tk.Text(self.info_frame, height=10, width=35, 
                                font=('Consolas', 9))
        self.info_text.pack(fill='x', padx=5, pady=5)
        self.info_text.config(state='disabled')
        
        return panel
    
    def _create_right_panel(self) -> ttk.Frame:
        """Create right visualization panel."""
        panel = ttk.Frame(self.main_paned)
        
        # Notebook for different plot views
        self.plot_notebook = ttk.Notebook(panel)
        self.plot_notebook.pack(fill='both', expand=True)
        
        # Tab 1: Overview plots
        self.overview_tab = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.overview_tab, text='信道概览')
        
        self.overview_fig = Figure(figsize=(12, 9), dpi=100)
        self.overview_canvas = FigureCanvasTkAgg(self.overview_fig, self.overview_tab)
        self.overview_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        toolbar_frame = ttk.Frame(self.overview_tab)
        toolbar_frame.pack(fill='x')
        NavigationToolbar2Tk(self.overview_canvas, toolbar_frame)
        
        # Tab 2: Statistics
        self.stats_tab = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.stats_tab, text='统计分析')
        
        self.stats_fig = Figure(figsize=(12, 9), dpi=100)
        self.stats_canvas = FigureCanvasTkAgg(self.stats_fig, self.stats_tab)
        self.stats_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        toolbar_frame2 = ttk.Frame(self.stats_tab)
        toolbar_frame2.pack(fill='x')
        NavigationToolbar2Tk(self.stats_canvas, toolbar_frame2)
        
        # Tab 3: Distribution fitting
        self.dist_tab = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.dist_tab, text='分布检验')
        
        self.dist_fig = Figure(figsize=(12, 9), dpi=100)
        self.dist_canvas = FigureCanvasTkAgg(self.dist_fig, self.dist_tab)
        self.dist_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        toolbar_frame3 = ttk.Frame(self.dist_tab)
        toolbar_frame3.pack(fill='x')
        NavigationToolbar2Tk(self.dist_canvas, toolbar_frame3)
        
        return panel
    
    def _create_status_bar(self):
        """Create status bar at bottom."""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(fill='x', side='bottom')
        
        self.status_label = ttk.Label(self.status_bar, text="就绪", 
                                     style='Status.TLabel')
        self.status_label.pack(side='left', padx=10)
        
        self.progress_bar = ttk.Progressbar(self.status_bar, mode='indeterminate',
                                           length=150)
        self.progress_bar.pack(side='right', padx=10)
    
    def _on_channel_type_changed(self):
        """Handle channel type selection change."""
        channel_type = self.channel_type_var.get()
        self.param_panel.set_channel_type(channel_type)
        self._update_status(f"已选择 {channel_type} 信道")
    
    def _on_params_changed(self):
        """Handle parameter change."""
        pass  # Can add real-time validation here
    
    def _generate_channel(self):
        """Generate channel based on current settings."""
        try:
            self._update_status("正在生成信道...")
            self.progress_bar.start()
            self.root.update()
            
            params = self.param_panel.get_parameters()
            channel_type = self.channel_type_var.get()
            
            sample_rate = params.get('sample_rate', 1e6)
            num_samples = int(params.get('num_samples', 10000))
            
            # Create channel based on type
            if channel_type == 'AWGN':
                self.current_channel = AWGNChannel(
                    snr_db=params.get('snr_db', 10.0),
                    sample_rate=sample_rate,
                    num_samples=num_samples
                )
            elif channel_type == 'Rayleigh':
                self.current_channel = RayleighChannel(
                    doppler_freq=params.get('doppler_freq', 100.0),
                    sample_rate=sample_rate,
                    num_samples=num_samples
                )
            elif channel_type == 'Rice':
                self.current_channel = RiceChannel(
                    k_factor=params.get('k_factor', 3.0),
                    doppler_freq=params.get('doppler_freq', 100.0),
                    sample_rate=sample_rate,
                    num_samples=num_samples
                )
            elif channel_type == 'Multipath':
                self.current_channel = MultipathChannel(
                    num_paths=int(params.get('num_paths', 4)),
                    doppler_freqs=params.get('multipath_doppler', 100.0),
                    rms_delay_spread=params.get('rms_delay_spread', 1e-6),
                    sample_rate=sample_rate,
                    num_samples=num_samples,
                    pdp_type=params.get('pdp_type', 'exponential')
                )
            
            # Generate channel response
            response = self.current_channel.generate()
            self.channel_data = response.impulse_response
            
            # Plot overview
            self._plot_overview()
            
            # Update info panel
            self._update_info_panel()
            
            self.progress_bar.stop()
            self._update_status(f"{channel_type} 信道生成完成")
            
        except Exception as e:
            self.progress_bar.stop()
            messagebox.showerror("错误", f"生成信道时出错: {str(e)}")
            self._update_status("生成失败")
    
    def _plot_overview(self):
        """Plot channel overview."""
        if self.channel_data is None:
            return
        
        self.overview_fig.clear()
        
        params = self.param_panel.get_parameters()
        sample_rate = params.get('sample_rate', 1e6)
        
        h = self.channel_data
        
        # Flatten for 1D plots if multipath
        if h.ndim == 2:
            h_flat = h[:, 0] if h.shape[1] > 0 else h.flatten()
        else:
            h_flat = h.flatten()
        
        time = np.arange(len(h_flat)) / sample_rate
        
        # 2x2 subplot layout
        ax1 = self.overview_fig.add_subplot(2, 2, 1)
        magnitude_db = 20 * np.log10(np.abs(h_flat) + 1e-10)
        ax1.plot(time * 1e3, magnitude_db, 'b-', linewidth=0.8)
        ax1.set_xlabel('时间 (ms)')
        ax1.set_ylabel('|h(t)| (dB)')
        ax1.set_title('信道幅度')
        ax1.grid(True, alpha=0.3)
        
        # Phase
        ax2 = self.overview_fig.add_subplot(2, 2, 2)
        phase = np.angle(h_flat)
        ax2.plot(time * 1e3, np.unwrap(phase) * 180 / np.pi, 'r-', linewidth=0.8)
        ax2.set_xlabel('时间 (ms)')
        ax2.set_ylabel('相位 (度)')
        ax2.set_title('信道相位')
        ax2.grid(True, alpha=0.3)
        
        # Frequency response
        ax3 = self.overview_fig.add_subplot(2, 2, 3)
        freq_resp = ChannelStatistics.compute_frequency_response(self.channel_data, sample_rate)
        ax3.plot(freq_resp.frequencies / 1e6, freq_resp.magnitude_db, 'g-', linewidth=1)
        ax3.set_xlabel('频率 (MHz)')
        ax3.set_ylabel('|H(f)| (dB)')
        ax3.set_title('频率响应')
        ax3.grid(True, alpha=0.3)
        
        # Histogram
        ax4 = self.overview_fig.add_subplot(2, 2, 4)
        ax4.hist(np.abs(h_flat), bins=50, density=True, alpha=0.7, color='steelblue')
        ax4.set_xlabel('幅度')
        ax4.set_ylabel('概率密度')
        ax4.set_title('幅度分布')
        ax4.grid(True, alpha=0.3)
        
        self.overview_fig.tight_layout()
        self.overview_canvas.draw()
    
    def _run_analysis(self):
        """Run statistical analysis on channel data."""
        if self.channel_data is None:
            messagebox.showwarning("警告", "请先生成信道")
            return
        
        try:
            self._update_status("正在进行统计分析...")
            self.progress_bar.start()
            self.root.update()
            
            params = self.param_panel.get_parameters()
            sample_rate = params.get('sample_rate', 1e6)
            doppler_freq = params.get('doppler_freq', params.get('multipath_doppler', 100.0))
            
            h = self.channel_data
            if h.ndim == 2:
                h_flat = h[:, 0] if h.shape[1] > 0 else h.flatten()
            else:
                h_flat = h.flatten()
            
            # Compute statistics
            pdp = ChannelStatistics.compute_pdp(self.channel_data, sample_rate)
            autocorr = ChannelStatistics.compute_autocorrelation(h_flat, sample_rate)
            freq_resp = ChannelStatistics.compute_frequency_response(self.channel_data, sample_rate)
            doppler_freq_axis, doppler_spectrum = ChannelStatistics.compute_doppler_spectrum(h_flat, sample_rate)
            
            self.analysis_results = {
                'pdp': pdp,
                'autocorr': autocorr,
                'freq_resp': freq_resp,
                'doppler_spectrum': (doppler_freq_axis, doppler_spectrum)
            }
            
            # Plot statistics
            self.stats_fig.clear()
            
            ax1 = self.stats_fig.add_subplot(2, 2, 1)
            ChannelPlotter.plot_pdp(pdp, ax=ax1, title='功率延迟分布 (PDP)')
            
            ax2 = self.stats_fig.add_subplot(2, 2, 2)
            ChannelPlotter.plot_autocorrelation(autocorr, ax=ax2, title='时间自相关函数')
            
            ax3 = self.stats_fig.add_subplot(2, 2, 3)
            ChannelPlotter.plot_frequency_response(
                freq_resp.frequencies, freq_resp.magnitude_db, ax=ax3, title='频率响应'
            )
            
            ax4 = self.stats_fig.add_subplot(2, 2, 4)
            ChannelPlotter.plot_doppler_spectrum(
                doppler_freq_axis, doppler_spectrum, doppler_freq, ax=ax4, title='多普勒功率谱'
            )
            
            self.stats_fig.tight_layout()
            self.stats_canvas.draw()
            
            # Switch to stats tab
            self.plot_notebook.select(self.stats_tab)
            
            self.progress_bar.stop()
            self._update_status("统计分析完成")
            
        except Exception as e:
            self.progress_bar.stop()
            messagebox.showerror("错误", f"分析时出错: {str(e)}")
    
    def _run_distribution_test(self):
        """Run distribution fitting test."""
        if self.channel_data is None:
            messagebox.showwarning("警告", "请先生成信道")
            return
        
        try:
            self._update_status("正在进行分布检验...")
            self.progress_bar.start()
            self.root.update()
            
            h = self.channel_data
            if h.ndim == 2:
                h_flat = h[:, 0] if h.shape[1] > 0 else h.flatten()
            else:
                h_flat = h.flatten()
            
            # Fit distributions
            rayleigh_fit = DistributionFitter.fit_rayleigh(h_flat)
            rice_fit = DistributionFitter.fit_rice(h_flat)
            gaussian_fit = DistributionFitter.fit_gaussian(h_flat.real)
            
            best_name, best_fit = DistributionFitter.find_best_fit(h_flat)
            
            # Generate report
            report = DistributionFitter.generate_report(h_flat)
            
            # Plot distribution results
            self.dist_fig.clear()
            
            # Histogram with fits
            ax1 = self.dist_fig.add_subplot(2, 2, 1)
            ChannelPlotter.plot_distribution_histogram(h_flat, rayleigh_fit, ax=ax1,
                                                       title='瑞利分布拟合')
            
            ax2 = self.dist_fig.add_subplot(2, 2, 2)
            ChannelPlotter.plot_distribution_histogram(h_flat, rice_fit, ax=ax2,
                                                       title='莱斯分布拟合')
            
            # Q-Q plots
            ax3 = self.dist_fig.add_subplot(2, 2, 3)
            theoretical, observed = DistributionFitter.generate_qq_data(h_flat, 'rayleigh')
            ChannelPlotter.plot_qq(theoretical, observed, 'Rayleigh', ax=ax3,
                                  title='瑞利 Q-Q 图')
            
            ax4 = self.dist_fig.add_subplot(2, 2, 4)
            theoretical, observed = DistributionFitter.generate_qq_data(h_flat, 'rice')
            ChannelPlotter.plot_qq(theoretical, observed, 'Rice', ax=ax4,
                                  title='莱斯 Q-Q 图')
            
            self.dist_fig.tight_layout()
            self.dist_canvas.draw()
            
            # Show report in info panel
            self.info_text.config(state='normal')
            self.info_text.delete('1.0', tk.END)
            self.info_text.insert('1.0', report)
            self.info_text.config(state='disabled')
            
            # Switch to distribution tab
            self.plot_notebook.select(self.dist_tab)
            
            self.progress_bar.stop()
            self._update_status(f"分布检验完成 - 最佳拟合: {best_name}")
            
        except Exception as e:
            self.progress_bar.stop()
            messagebox.showerror("错误", f"分布检验时出错: {str(e)}")
    
    def _update_info_panel(self):
        """Update channel information panel."""
        if self.current_channel is None or self.channel_data is None:
            return
        
        params = self.param_panel.get_parameters()
        sample_rate = params.get('sample_rate', 1e6)
        
        h = self.channel_data
        if h.ndim == 2:
            h_flat = h[:, 0] if h.shape[1] > 0 else h.flatten()
        else:
            h_flat = h.flatten()
        
        # Compute quick stats
        summary = DistributionFitter.compute_summary(h_flat)
        
        info_lines = [
            f"信道类型: {self.channel_type_var.get()}",
            f"采样率: {sample_rate:.2e} Hz",
            f"样本数: {len(h_flat)}",
            "-" * 30,
            f"均值: {summary.mean:.6f}",
            f"标准差: {summary.std:.6f}",
            f"方差: {summary.variance:.6f}",
            f"偏度: {summary.skewness:.4f}",
            f"峰度: {summary.kurtosis:.4f}",
            f"最小值: {summary.min_val:.6f}",
            f"最大值: {summary.max_val:.6f}",
        ]
        
        # Add channel-specific info
        if hasattr(self.current_channel, 'get_coherence_time'):
            tc = self.current_channel.get_coherence_time()
            info_lines.append(f"相干时间: {tc*1e3:.4f} ms")
        
        if hasattr(self.current_channel, 'get_coherence_bandwidth'):
            bc = self.current_channel.get_coherence_bandwidth()
            info_lines.append(f"相干带宽: {bc/1e3:.2f} kHz")
        
        self.info_text.config(state='normal')
        self.info_text.delete('1.0', tk.END)
        self.info_text.insert('1.0', '\n'.join(info_lines))
        self.info_text.config(state='disabled')
    
    def _show_export_dialog(self):
        """Show export dialog."""
        if self.channel_data is None:
            messagebox.showwarning("警告", "请先生成信道")
            return
        
        ExportDialog(self.root, self._do_export)
    
    def _do_export(self, fmt: str, filepath: str, options: dict):
        """
        Perform data export.
        
        Args:
            fmt: Export format
            filepath: Output path
            options: Export options
        """
        try:
            self._update_status(f"正在导出到 {fmt}...")
            
            params = self.param_panel.get_parameters()
            config = {
                'channel_type': self.channel_type_var.get(),
                'sample_rate': params.get('sample_rate', 1e6),
                'num_samples': int(params.get('num_samples', 10000)),
                **params
            }
            
            if fmt == 'package':
                # Full export package
                report = DistributionFitter.generate_report(self.channel_data)
                ChannelExporter.create_export_package(
                    data=self.channel_data,
                    config=config,
                    analysis_results=self.analysis_results,
                    report=report,
                    output_dir=filepath
                )
            elif fmt == 'npy':
                ChannelExporter.export_to_numpy(self.channel_data, filepath)
            elif fmt == 'csv':
                ChannelExporter.export_to_csv(
                    self.channel_data, filepath, 
                    sample_rate=config['sample_rate']
                )
            elif fmt == 'mat':
                mat_data = {
                    'h': self.channel_data,
                    'fs': config['sample_rate'],
                    'channel_type': config['channel_type']
                }
                ChannelExporter.export_to_mat(mat_data, filepath)
            elif fmt == 'json':
                ChannelExporter.export_to_json(config, filepath)
            
            self._update_status(f"导出成功: {filepath}")
            messagebox.showinfo("成功", f"数据已导出到:\n{filepath}")
            
        except Exception as e:
            messagebox.showerror("错误", f"导出失败: {str(e)}")
    
    def _save_config(self):
        """Save current configuration to file."""
        filepath = filedialog.asksaveasfilename(
            title="保存配置",
            filetypes=[('JSON文件', '*.json')],
            defaultextension='.json'
        )
        if filepath:
            params = self.param_panel.get_parameters()
            config = {
                'channel_type': self.channel_type_var.get(),
                **params
            }
            ChannelExporter.export_to_json(config, filepath)
            self._update_status(f"配置已保存: {filepath}")
    
    def _load_config(self):
        """Load configuration from file."""
        filepath = filedialog.askopenfilename(
            title="载入配置",
            filetypes=[('JSON文件', '*.json')]
        )
        if filepath:
            try:
                import json
                with open(filepath) as f:
                    config = json.load(f)
                
                if 'channel_type' in config:
                    self.channel_type_var.set(config['channel_type'])
                    self._on_channel_type_changed()
                
                self.param_panel.set_parameters(config)
                self._update_status(f"配置已载入: {filepath}")
                
            except Exception as e:
                messagebox.showerror("错误", f"载入配置失败: {str(e)}")
    
    def _refresh_plots(self):
        """Refresh all plots."""
        if self.channel_data is not None:
            self._plot_overview()
            if self.analysis_results:
                self._run_analysis()
    
    def _reset_view(self):
        """Reset view to initial state."""
        self.overview_fig.clear()
        self.stats_fig.clear()
        self.dist_fig.clear()
        self.overview_canvas.draw()
        self.stats_canvas.draw()
        self.dist_canvas.draw()
        
        self.info_text.config(state='normal')
        self.info_text.delete('1.0', tk.END)
        self.info_text.config(state='disabled')
        
        self._update_status("视图已重置")
    
    def _show_about(self):
        """Show about dialog."""
        about_text = """无线信道建模与仿真软件
Wireless Channel Simulator

版本: 1.0.0

功能:
• AWGN/瑞利/莱斯/多径信道生成
• 功率延迟分布 (PDP) 分析
• 自相关函数计算
• 频率响应分析
• 分布拟合检验 (K-S 检验)
• 多格式数据导出

© 2024"""
        messagebox.showinfo("关于", about_text)
    
    def _update_status(self, message: str):
        """Update status bar message."""
        self.status_label.config(text=message)
        self.root.update()


def run_app():
    """Run the application."""
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()


if __name__ == '__main__':
    run_app()
