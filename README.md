# Wireless Channel Simulator

一个基于 Python 的无线信道建模与参数化仿真项目，提供 AWGN、Rayleigh、Rice 和多径信道模型，并配套图形化界面、统计分析与数据导出能力。

## 功能概览

- **信道模型**
  - AWGN（加性白高斯噪声）
  - Rayleigh（瑞利衰落）
  - Rice（莱斯衰落，含 K 因子）
  - Multipath（时变多径信道，支持多种 PDP）
- **可配置参数**
  - 采样率、样本数、SNR、多普勒频移、K 因子、路径数、RMS 延迟扩展等
- **分析与可视化**
  - 时域/频域响应、统计量分析、分布拟合
- **数据导出**
  - 支持 `.npy`、`.csv`、`.mat`、`.json` 以及打包导出
- **桌面 GUI**
  - 基于 Tkinter + Matplotlib 的可交互仿真界面

## 项目结构

```text
Wireless/
├── main.py                    # 程序入口（启动 GUI）
├── channels/                  # 各类信道模型
│   ├── base.py
│   ├── awgn.py
│   ├── rayleigh.py
│   ├── rice.py
│   └── multipath.py
├── config/                    # 配置与枚举
├── analysis/                  # 统计分析与分布拟合
├── visualization/             # 绘图模块
├── export/                    # 导出模块
├── gui/                       # 图形界面
└── requirements.txt
```

## 环境要求

- Python 3.7+
- 建议系统已具备图形环境（运行 GUI 时需要）

依赖如下：

- numpy>=1.21.0
- scipy>=1.7.0
- matplotlib>=3.4.0

## 安装

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## 运行方式

### 1) 启动图形界面

```bash
python main.py
```

启动后可在左侧选择信道类型并配置参数，右侧查看仿真图表与分析结果。

### 2) 作为 Python 模块调用（示例）

```python
import numpy as np
from channels import RayleighChannel

# 创建 Rayleigh 信道
ch = RayleighChannel(
    doppler_freq=50.0,
    sample_rate=1e6,
    num_samples=4096,
)

# 生成信道响应
resp = ch.generate()
print(resp.impulse_response.shape)

# 将信道作用到输入信号
x = np.ones(4096, dtype=complex)
y = ch.apply(x)
print(y[:5])
```

## 多径信道说明

`MultipathChannel` 支持以下功率延迟分布（`pdp_type`）：

- `exponential`
- `uniform`
- `itu_ped_a`
- `itu_veh_a`

你也可以在配置中扩展为自定义 PDP（见 `config/channel_config.py` 的 `PDPType.CUSTOM`）。

## 导出能力

导出模块 `export/` 可将仿真数据与配置保存为常见格式，适合后续在 NumPy / MATLAB / 数据分析流程中处理。

## 常见问题

1. **GUI 无法启动 / Tk 报错**
   - 请确认系统已安装 Tk 组件并可用图形显示环境。
2. **导出 `.mat` 失败**
   - 请确认 `scipy` 安装正常。
3. **中文显示异常**
   - 可根据系统字体情况调整 GUI 或 Matplotlib 字体配置。

## 许可

当前仓库未显式提供 LICENSE 文件。如需开源分发，请补充许可证声明。
