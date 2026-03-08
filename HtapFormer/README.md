# HtapFormer

## 1. Environment Setup

Create a conda environment:

```bash
conda env create -f environment.yml
conda activate htapformer
```

## 2. Data Preparation

 Prepare your data as follow example data files:
- `data/plan/TPCH-10.csv`
- `data/plan/TPCH-5.csv`
- `data/plan/hybench5.csv`
- `data/plan/hybench10.csv`

## 3. Model Training

Run the main training script:

```bash
python HtapFormer.py
```

## 4. Write Sensitivity Coefficient Collection (Optional)

To customize write sensitivity coefficients (α_i, α_u, α_d), run:

```bash
# Online mode: Connect to database and collect data
python collect_write_sensitivity.py --host 127.0.0.1 --port 5432 --dbname YOURDB --user YOURUSERNAME --password YOURPASSWORD

# Offline mode: Use existing CSV file
python collect_write_sensitivity.py --from-csv-only
```

Results will be saved to `checkpoints/write_sensitivity.json`. If the file does not exist, the model will use default values (α=1.0).

## 5.File Structure

```
HtapFormer/
├── HtapFormer.py              # Main training script
├── collect_write_sensitivity.py  # Write sensitivity collection script
├── model/                     # Model implementation
│   ├── model.py              # HtapFormer model definition
│   ├── htap_bias.py          # HTAP-Bias module
│   ├── dataset.py            # Dataset processing
│   ├── trainer.py            # Trainer
│   └── ...
└── 
```

The "data" and "checkpoint" folders need to be prepared in advance, with the following structure:

```
├── data/                      # Data directory
│   └── plan/                  # Query plan data
├── checkpoints/               # Model checkpoints
│   ├── encoding.pt           # Encoding dictionary
│   ├── cost_model.pt         # Pre-trained model (optional)
│   └── write_sensitivity.json # Write sensitivity coefficients
```

