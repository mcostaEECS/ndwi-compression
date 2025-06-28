# Adaptive NDWI-Based Compression with Seasonal Change Detection

[![IEEE](https://img.shields.io/badge/IEEE-GRSL-8A2BE2)](https://ieeexplore.ieee.org/document/XXXXXXX)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-green)](LICENSE)

Official implementation for the paper:  
**"Adaptive Compression Rate with Seasonal Change Detection on NDWI Composition"** (Submitted to IEEE GRSL/TGRS)

> On-the-fly compression method for satellite/UAV platforms that dynamically adapts compression ratios using NDWI-based change detection and rate-distortion theory. Achieves **>5:1 compression** on water/ice regions while preserving critical features.

## Key Features
- 🌊 **NDWI-driven compression** for water/ice regions
- ❄️ **Seasonal adaptation** (summer/winter modes)
- ⚡ **Hardware-efficient** implementation (ARM Cortex-R5 compatible)
- 📊 Rate-distortion optimized thresholds
- 🛰️ Validated on PlanetScope satellite imagery

## Repository Structure

├── analysis/ # Jupyter notebooks for result analysis
├── data/ # Satellite imagery datasets
│ ├── Scene_sea/ # Baltic Sea surface scenes
│ ├── Scene_summer/ # Summer acquisitions (Vaxholmen)
│ └── Scene_winter/ # Winter acquisitions (Vaxholmen)
├── Metadata/ # Scene metadata
├── temperature/ # Temperature data (exogenous variable)
├── wind/ # Wind speed data (exogenous variable)
├── main_simulation/ # Core compression algorithms
│ ├── AnomalyComp.py # Main compression module
│ ├── AnomalyGrid.py # Grid-based change detection
│ └── KTH-Compress.py # CLI interface for compression
├── results/ # Precomputed outputs (figures, metrics)
├── tools/ # Utility scripts
│ ├── csvReadTEMP.py # Temperature data loader
│ ├── csvReadwind.py # Wind data loader
│ ├── getDataset.py # Data fetcher (Mimir's Well API)
│ └── getDataset2.py # Alternative data fetcher
├── Data_Holo.csv # Sandham (sea region) metadata
├── Data_Vaxholm.csv # Vaxholmen region metadata
└── requirements.txt # Python dependencies


## Getting Started

### 1. Installation
```bash
git clone https://github.com/yourusername/ndwi-compression.git
cd ndwi-compression
pip install -r requirements.txt 
```
### 2. Data Preparation

Download PlanetScope datasets:
```bash
python tools/getDataset.py --region vaxholmen --season summer --output data/Scene_summer
```

### 3. Run Compression

Process a test scene with adaptive compression:
```bash
python main_simulation/KTH-Compress.py \
  --input data/Scene_summer/20230615.tif \
  --output results/compressed \
  --season summer \
  --quality 95 \
  --bpp_target 8
```
### 4. Reproduce Paper Results
```bash
# Full experimental pipeline
python scripts/run_paper_experiments.py \
  --regions vaxholmen sandham \
  --seasons summer winter
```

Hardware Emulation
To benchmark on ARM Cortex-R5:

# Install QEMU
sudo apt install qemu-system-arm

# Cross-compile
arm-none-eabi-gcc -mcpu=cortex-r5 -O3 main_simulation/AnomalyComp.c -o firmware.bin

# Run emulation
qemu-system-arm -M xilinx-zynqmp -cpu cortex-r5f -nographic -kernel firmware.bin

### Citation

bibtex
@article{costa2024adaptive,
  title={Adaptive Compression Rate with Seasonal Change Detection on NDWI Composition},
  author={Costa, Marcello and Sander, Ingo and Soderquist, Ingemar and Pinho, Marcelo and Dammert, Patrik and Fuglesang, Christer},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2024},
  publisher={IEEE}
}

### License
BSD 3-Clause License. See LICENSE for details.


## Recommended Additions

1. Create these supporting files:
   - `requirements.txt` with:
     ```
     numpy>=1.21
     rasterio>=1.3
     scipy>=1.9
     pandas>=1.5
     matplotlib>=3.6
     ```
   - `LICENSE` (BSD 3-Clause recommended)
   - `scripts/run_paper_experiments.py` to automate reproduction

2. For hardware emulation:
   - Add `Dockerfile` with ARM toolchain
   - Include sample outputs in `results/hardware/`

3. Enhance documentation:
   - Add Jupyter notebooks in `analysis/` showing:
     - NDWI visualization
     - Compression quality metrics
     - Seasonal comparisons

4. Continuous Integration:
   ```yml
   # .github/workflows/ci.yml
   name: CI
   on: [push]
   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
       - uses: actions/checkout@v4
       - name: Setup Python
         uses: actions/setup-python@v4
         with: {python-version: '3.10'}
       - run: pip install -r requirements.txt
       - run: python -m pytest tests/
