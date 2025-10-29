"""
HtapFormer Training Script

Based on QueryFormer's TrainingV1.py, adapted for HtapFormer model.
Main changes:
1. Use HtapFormer model instead of QueryFormer
2. Add HTAP feature extraction and data loading
3. Support htap_lambda parameter tuning
4. Maintain compatibility with QueryFormer training workflow

Usage:
    python train_htapformer.py

Notes:
- If real HTAP features are not available, mock data can be used (see below)
- Hyperparameters can be adjusted by modifying the Args class
- Training results are saved in ./results/htapformer/cost/ directory
"""

# %%
import numpy as np
import os
import torch
import torch.nn as nn
import time
import pandas as pd
from scipy.stats import pearsonr

# %%
from model.htap_utils import Normalizer, seed_everything  # Integrated into htap_utils
from model.database_util import get_hist_file, get_job_table_sample, collator
from model.htapformer_model import HtapFormer  # Use HtapFormer
from model.database_util import Encoding
from model.dataset import PlanTreeDataset
# Use HTAP trainer
from model.htap_trainer import train_htapformer, eval_workload_htapformer

# %%
data_path = './data/imdb/'

# %%
class Args:
    """
    Training hyperparameter configuration
    
    HtapFormer-specific parameters:
    - htap_lambda: HTAP bias weight (key parameter!)
    - storage_feature_dim: storage feature dimension
    - operator_feature_dim: operator feature dimension
    - use_htap_features: whether to use HTAP features (False equals QueryFormer)
    """
    # Basic hyperparameters (same as QueryFormer)
    bs = 128                    # batch size
    lr = 0.001                  # learning rate
    epochs = 100                # training epochs
    clip_size = 50              # gradient clipping
    
    # Model architecture parameters
    embed_size = 64             # embedding dimension
    pred_hid = 128              # prediction head hidden size
    ffn_dim = 128               # FFN dimension
    head_size = 12              # number of attention heads
    n_layers = 8                # number of encoder layers
    dropout = 0.1               # dropout rate
    
    # HtapFormer-specific parameters
    htap_lambda =           # HTAP bias weight λ (key parameter!)
    storage_feature_dim = 8     # storage mode feature dimension
    operator_feature_dim = 8    # operator type feature dimension
    use_htap_features = True    # whether to use HTAP features
    
    # Training configuration
    sch_decay = 0.6             # scheduler decay rate
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Path configuration
    newpath = './results/htapformer/cost/'
    to_predict = 'cost'

args = Args()

# Create results directory
if not os.path.exists(args.newpath):
    os.makedirs(args.newpath)

print("=" * 60)
print("HtapFormer Training Configuration")
print("=" * 60)
print(f"Device: {args.device}")
print(f"Batch size: {args.bs}")
print(f"Learning rate: {args.lr}")
print(f"Epochs: {args.epochs}")
print(f"Embed size: {args.embed_size}")
print(f"FFN dim: {args.ffn_dim}")
print(f"Head size: {args.head_size}")
print(f"Num layers: {args.n_layers}")
print(f"\n🌟 HtapFormer Parameters:")
print(f"  - HTAP lambda (λ): {args.htap_lambda}")
print(f"  - Use HTAP features: {args.use_htap_features}")
print(f"  - Storage feature dim: {args.storage_feature_dim}")
print(f"  - Operator feature dim: {args.operator_feature_dim}")
print("=" * 60 + "\n")

# %%
print("Loading histogram and normalization...")
hist_file = get_hist_file(data_path + 'histogram_string.csv')
cost_norm = Normalizer(-3.61192, 12.290855)
card_norm = Normalizer(1, 100)

# %%
print("Loading encoding and checkpoint...")
encoding_ckpt = torch.load('checkpoints/encoding.pt')
encoding = encoding_ckpt['encoding']

# Optional: load pre-trained QueryFormer weights
# checkpoint = torch.load('checkpoints/cost_model.pt', map_location='cpu')

# %%
print("Setting random seed...")
seed_everything()

# %%
print("Creating HtapFormer model...")
model = HtapFormer(
    emb_size=args.embed_size,
    ffn_dim=args.ffn_dim,
    head_size=args.head_size,
    dropout=args.dropout,
    n_layers=args.n_layers,
    use_sample=True,
    use_hist=True,
    pred_hid=args.pred_hid,
    # HtapFormer-specific parameters
    storage_feature_dim=args.storage_feature_dim,
    operator_feature_dim=args.operator_feature_dim,
    htap_lambda=args.htap_lambda
)

print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# %%
_ = model.to(args.device)
print(f"Model moved to {args.device}\n")

# %%
to_predict = 'cost'

# %%
print("Loading training and validation data...")
imdb_path = './data/imdb/'

# Training data
dfs = []
print("Loading training data...")
# Note: Only load 2 files as an example, load more for full training
# For complete training, change range(2) to range(18)
for i in range(2):
    file = imdb_path + f'plan_and_cost/train_plan_part{i}.csv'
    df = pd.read_csv(file)
    dfs.append(df)
    print(f"  Loaded {file}: {len(df)} samples")

full_train_df = pd.concat(dfs)
print(f"Total training samples: {len(full_train_df)}")

# Validation data
val_dfs = []
print("\nLoading validation data...")
for i in range(18, 20):
    file = imdb_path + f'plan_and_cost/train_plan_part{i}.csv'
    df = pd.read_csv(file)
    val_dfs.append(df)
    print(f"  Loaded {file}: {len(df)} samples")

val_df = pd.concat(val_dfs)
print(f"Total validation samples: {len(val_df)}\n")

# %%
print("Loading table samples...")
table_sample = get_job_table_sample(imdb_path + 'train')

# %%
print("Creating datasets...")

# ==================== HTAP Feature Processing ====================
# Note: htap_trainer.py will automatically add mock HTAP features
# In actual use, you need to:
# 1. Create HTAPPlanTreeDataset class (refer to HTAP_DATA_GUIDE.md)
# 2. Extract real HTAP features from database or workload history
# 3. Replace create_mock_htap_data function in htap_trainer.py

if args.use_htap_features:
    print("\n Currently using mock HTAP features (auto-generated by htap_trainer.py)")
    print("   For actual use, you need to:")
    print("   1. Create HTAPPlanTreeDataset class")
    print("   2. Extract real storage types, operation frequencies, etc.")
    print("   3. Refer to HTAP_DATA_GUIDE.md for detailed guide\n")
    
train_ds = PlanTreeDataset(
    full_train_df, None, encoding, hist_file, 
    card_norm, cost_norm, to_predict, table_sample
)

val_ds = PlanTreeDataset(
    val_df, None, encoding, hist_file, 
    card_norm, cost_norm, to_predict, table_sample
)

print(f"Training dataset size: {len(train_ds)}")
print(f"Validation dataset size: {len(val_ds)}")


# %%
print("\n" + "=" * 60)
print("Starting Training")
print("=" * 60 + "\n")

crit = nn.MSELoss()

try:
    # Use HTAP trainer
    model, best_path = train_htapformer(
        model, train_ds, val_ds, crit, cost_norm, args,
        use_htap=args.use_htap_features
    )
    print(f"\n✅ Training completed!")
    print(f"Best model saved at: {best_path}")
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    print("Please check the error message above.")
    raise

# %%
print("\n" + "=" * 60)
print("Evaluation on Workloads")
print("=" * 60 + "\n")

methods = {
    'get_sample': get_job_table_sample,
    'encoding': encoding,
    'cost_norm': cost_norm,
    'hist_file': hist_file,
    'model': model,
    'device': args.device,
    'bs': 512,
}

# %%
print("Evaluating on job-light workload...")
try:
    job_light_result, _ = eval_workload_htapformer(
        'job-light', methods, use_htap=args.use_htap_features
    )
    print(f"job-light evaluation completed")
    print(f"  Q-Median: {job_light_result['q_median']:.2f}")
    print(f"  Q-Mean: {job_light_result['q_mean']:.2f}")
    print(f"  Correlation: {job_light_result['corr']:.4f}")
except Exception as e:
    print(f"job-light evaluation failed: {e}")

# %%
print("\nEvaluating on synthetic workload...")
try:
    synthetic_result, _ = eval_workload_htapformer(
        'synthetic', methods, use_htap=args.use_htap_features
    )
    print(f"synthetic evaluation completed")
    print(f"  Q-Median: {synthetic_result['q_median']:.2f}")
    print(f"  Q-Mean: {synthetic_result['q_mean']:.2f}")
    print(f"  Correlation: {synthetic_result['corr']:.4f}")
except Exception as e:
    print(f"synthetic evaluation failed: {e}")

# %%
print("\n" + "=" * 60)
print("Training and Evaluation Completed!")
print("=" * 60)
print(f"\nModel save path: {args.newpath}")
print(f"Best model: {best_path if 'best_path' in locals() else 'N/A'}")
print("\nNext steps:")
print("  1. View training logs: {}/log.txt".format(args.newpath))
print("  2. Tune lambda parameter: modify Args.htap_lambda")
print("  3. Use real HTAP features: refer to HTAP_DATA_GUIDE.md")
print("  4. Run ablation studies: compare effects of different lambda values")

# %%
# Save training configuration
import json
config_save_path = os.path.join(args.newpath, 'config.json')
config_dict = {
    'bs': args.bs,
    'lr': args.lr,
    'epochs': args.epochs,
    'embed_size': args.embed_size,
    'ffn_dim': args.ffn_dim,
    'head_size': args.head_size,
    'n_layers': args.n_layers,
    'htap_lambda': args.htap_lambda,
    'storage_feature_dim': args.storage_feature_dim,
    'operator_feature_dim': args.operator_feature_dim,
    'use_htap_features': args.use_htap_features,
    'device': args.device,
}

with open(config_save_path, 'w') as f:
    json.dump(config_dict, f, indent=2)

print(f"\nTraining configuration saved to: {config_save_path}")
