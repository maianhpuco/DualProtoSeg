# DualProtoSeg: Simple and Efficient Design with Text- and Image-Guided Prototype Learning for Weakly Supervised Histopathology Image Segmentation  

**Quick setup**

- 1) Download the BCSS dataset and place it under the project `data/` folder.
  - Example layout (common):
    - `data/train/`
    - `data/val/`
    - `data/test/`
  - Adjust dataset paths in the config if your layout differs.

- 2) Install CONCH (project-specific):
  - If a pip package is available, you can try:

```bash
# inside a venv (recommended)
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install open_clip_torch
```

  - Or install the CONCH repo (if provided by your team / HF):

```bash
# example - replace with the actual CONCH repo URL if needed
git clone https://github.com/MahmoodLab/conch.git
cd conch
pip install -e .
```

- 3) Install Python dependencies from `requirements.txt`:

```bash
# activate your virtualenv first
. .venv/bin/activate
pip install -r requirements.txt
```

Note: For `torch`, pick the wheel that matches your CUDA version. Example (change `cu121` to your CUDA version):

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
```

- 4) Run training with the Makefile

```bash
# run with defaults: CONFIG=config.yaml, GPU=0
make run

# or explicitly
make run CONFIG=config.yaml GPU=0
```

The script will default to saving outputs under a `runs/` folder if the config path has no directory component.

**Notes & tips**
- If your `config.yaml` points to a CONCH checkpoint (e.g., `clip.checkpoint_path`), ensure that file path exists.
- To resume training from a checkpoint, pass `--resume path/to/checkpoint.pth` to `train.py` (edit `Makefile` or run `python3 train.py ...`).


## Acknowledgment
Parts of the `utils/` folder were inspired by the excellent PBIP implementation by  
[QingchenTang](https://github.com/QingchenTang/PBIP).  
We sincerely thank the authors for making their code publicly available.
 
