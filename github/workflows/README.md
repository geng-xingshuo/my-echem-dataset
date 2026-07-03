# my-echem-dataset

Repository to host synthetic electrochemical plots and tools to generate paired image/CSV datasets for chart digitization evaluation.

This repo includes:
- generate_echem_dataset.py : script to synthesize CV, charge/discharge, Nyquist, multi-line and scatter plots and save ground-truth CSVs.
- .github/workflows/generate_dataset.yml : GitHub Action to run the generator and commit outputs back to the repo (triggered on push or manually).

To generate dataset locally:

1. Create virtual env and install deps:

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: .\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install numpy matplotlib pandas pillow
