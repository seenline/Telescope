# Telescope

## HtapFormer

HtapFormer， a model for HTAP-aware query cost estimation using transformer-based models. 

### Learnable Parameters

The following parameters are learned during training:

| Unallocated Parameter | Location | Type | Description |
| --------- | -------- | ---- | ----------- |
| `alpha_insert` | model/htap_utils.py | Learnable | INSERT operation sensitivity coefficient |
| `alpha_update` | model/htap_utils.py | Learnable | UPDATE operation sensitivity coefficient |
| `alpha_delete` | model/htap_utils.py | Learnable | DELETE operation sensitivity coefficient |
| `storage_embed` | model/htap_utils.py | Learnable | Storage type embeddings (row/column/hybrid) |
| `query_type_embed` | model/htap_utils.py | Learnable | Query type embeddings (INSERT/UPDATE/DELETE/SELECT) |

### Hyperparameters

The following parameters are set as hyperparameters:

| Parameter | Location | Default | Type | Description |
| --------- | -------- | ------- | ---- | ----------- |
| `htap_lambda` | train_htapformer.py | None | Hyperparameter | HTAP bias weight (λ) in attention mechanism |

### Formula

Storage Bias: `v_j^s = α_i × cnt_i + α_u × cnt_u + α_d × cnt_d`

HTAP-Bias Attention: `A'_{ij} = (QK^T)/√d + b_tree + λ × b_HTAP(i,j)`

where `b_HTAP(i,j) = f_s(v_i^s, v_j^s) + f_o(v_i^o, v_j^o)`



## Project Structure

```
Telescope/
├── HtapFormer/              # HtapFormer implementation
│   ├── model/              # Model components
│   ├── train_htapformer.py # Training script
│   └── README.md           # Full documentation
└── README.md               # This file
```

## Quick Start

### 1. Install Dependencies
```bash
cd HtapFormer
pip install -r requirements.txt
```

### 2. Prepare Data
 prepare your own data following the structure:
```
data/imdb/
├── histogram_string.csv
├── plan_and_cost/
│   ├── train_plan_part0.csv
│   └── ...
└── workloads/
    ├── job-light.csv
    └── synthetic.csv

checkpoints/
└── encoding.pt
```

### 3. Train the Model
```bash
python train_htapformer.py
```

Default configuration:
- Batch size: 128
- Epochs: 100
- Learning rate: 0.001
- HTAP lambda: 0.1
- Training files: 2 (for quick test)

### 4. View Results
Training results saved to:
```
results/htapformer/cost/
├── model_epoch_XX.pt    # Best model checkpoint
├── log.txt              # Training log
└── config.json          # Training configuration
```

### 5. Modify Hyperparameters (Optional)
Edit `train_htapformer.py`:
```python
class Args:
    htap_lambda = 0.1  # Change HTAP bias weight
    epochs = 100       # Change number of epochs
    bs = 128          # Change batch size
```

For full training (18 files):
```python
# Line ~157 in train_htapformer.py
for i in range(18):  # Change from range(2) to range(18)
```

## Status

✅ Model implementation complete  
✅ Training pipeline functional  
⚠️ Currently using mock HTAP features (random data)  
🔜 TODO: Extract real HTAP features from database

---

**For detailed documentation, see [HtapFormer/README.md](../HtapFormer/README.md)**
