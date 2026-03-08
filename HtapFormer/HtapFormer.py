# %%
import numpy as np
import os
import torch
import torch.nn as nn
import time
import pandas as pd
from scipy.stats import pearsonr
import random
import json

# %%
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

if os.path.isdir(os.path.join(CURRENT_DIR, 'model')):
    PROJECT_ROOT = CURRENT_DIR
elif os.path.isdir(os.path.join(PARENT_DIR, 'model')):
    PROJECT_ROOT = PARENT_DIR
else:
    PROJECT_ROOT = CURRENT_DIR

for path in {CURRENT_DIR, PROJECT_ROOT}:
    if path not in sys.path:
        sys.path.insert(0, path)

from model.util import Normalizer  # noqa: E402
from model.database_util import collator  # noqa: E402
from model.model import HtapFormer  # noqa: E402
from model.database_util import Encoding  # noqa: E402
from model.dataset import PlanTreeDataset  # noqa: E402
from model.trainer import eval_workload, train  # noqa: E402

all_results = []

# ===================== Unified Configuration =====================
# All training and model parameters are configured in this section

# ========== Data Settings ==========
data_file = os.path.join(CURRENT_DIR, "data", "plan", "TPCH-10.csv")
train_ratio =   # Proportion of data used for training

# ========== Model Hyperparameters ==========
class Args:
    # Core training parameters
    bs = 
    lr = 
    epochs = 
    clip_size = 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    newpath = os.path.join(CURRENT_DIR, 'results', 'full', 'cost')
    to_predict = 'cost'
    
    # Encoder/decoder structure
    embed_size = 
    pred_hid = 
    ffn_dim = 
    head_size = 
    n_layers = 
    dropout = 
    sch_decay = 
    
    # HTAP-Bias parameters
    use_htap_bias = 
    lambda_scale = 
    learnable_weights = 
    

    # INSERT, UPDATE, DELETE, SELECT
    query_weight_init = [, , , ]
    # Scan, Join, Aggregate, GroupBy, Other
    operator_weight_init = [, , , , ]

args = Args()

all_df = pd.read_csv(data_file)
all_idx = list(range(len(all_df)))
random.shuffle(all_idx)
train_size = int(len(all_df) * train_ratio)
train_idx = all_idx[:train_size]
val_idx = all_idx[train_size:]
train_df = all_df.iloc[train_idx].reset_index(drop=True)
val_df = all_df.iloc[val_idx].reset_index(drop=True)
print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")

if not os.path.exists(args.newpath):
    os.makedirs(args.newpath)

execution_times = all_df['Execution Time'].values
cost_norm = Normalizer(
    mini=np.log(execution_times.min()),
    maxi=np.log(execution_times.max())
)
card_norm = Normalizer(1, 100)


encoding_ckpt = torch.load(os.path.join(CURRENT_DIR, 'checkpoints', 'encoding.pt'), weights_only=False)
encoding_dict = encoding_ckpt['encoding']
encoding = Encoding(
    encoding_dict['column_min_max_vals'],
    encoding_dict['col2idx'],
    encoding_dict['op2idx']
)
encoding.type2idx = encoding_dict['type2idx']
encoding.idx2type = encoding_dict['idx2type']
encoding.join2idx = encoding_dict['join2idx']
encoding.idx2join = encoding_dict['idx2join']
checkpoint = torch.load(os.path.join(PROJECT_ROOT, 'checkpoints', 'cost_model.pt'), map_location='cpu', weights_only=False)


def load_htap_alphas(alpha_path: str):
    """Load HTAP write-sensitivity coefficients (α_i, α_u, α_d)."""
    defaults = {"alpha_i": 1.0, "alpha_u": 1.0, "alpha_d": 1.0}
    if not os.path.exists(alpha_path):
        print(f"[WARN] Missing write-sensitivity file {alpha_path}; default α=1.0 is used")
        return defaults
    try:
        with open(alpha_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "alpha_i": float(data.get("alpha_i", defaults["alpha_i"])),
            "alpha_u": float(data.get("alpha_u", defaults["alpha_u"])),
            "alpha_d": float(data.get("alpha_d", defaults["alpha_d"])),
        }
    except Exception as exc:
        print(f"[WARN] Failed to load {alpha_path} ({exc}); default α=1.0 is used")
        return defaults

alpha_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'write_sensitivity.json')
htap_alphas = load_htap_alphas(alpha_path)

from model.util import seed_everything
seed_everything()

model = HtapFormer(
    emb_size=args.embed_size,
    ffn_dim=args.ffn_dim,
    head_size=args.head_size,
    dropout=args.dropout,
    n_layers=args.n_layers,
    pred_hid=args.pred_hid,
    joins=len(encoding.join2idx),
    use_htap_bias=args.use_htap_bias,
    lambda_scale=args.lambda_scale,
    alpha_i=htap_alphas["alpha_i"],
    alpha_u=htap_alphas["alpha_u"],
    alpha_d=htap_alphas["alpha_d"],
    learnable_weights=args.learnable_weights,
    query_weight_init=args.query_weight_init,
    operator_weight_init=args.operator_weight_init
)
_ = model.to(args.device)
to_predict = 'cost'


train_ds = PlanTreeDataset(train_df, None, encoding, None, card_norm, cost_norm, to_predict, None)
val_ds = PlanTreeDataset(val_df, None, encoding, None, card_norm, cost_norm, to_predict, None)

crit = nn.MSELoss()
model, best_path = train(model, train_ds, val_ds, crit, cost_norm, args)

from torch.utils.data import DataLoader
def test_collate(batch):
    dicts = [item[0] for item in batch]
    cost_labels = [item[1][0] for item in batch] 
    from model.database_util import collator
    batch_obj, _ = collator((dicts, cost_labels))
    return batch_obj, torch.stack(cost_labels)

val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=test_collate)

model.eval()
preds = []
labels = []
with torch.no_grad():
    for batch, y in val_loader:
        batch = batch.to(args.device)
        y = y.to(args.device)
        y_pred = model(batch)
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0] 
        preds.append(y_pred.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())

preds = np.concatenate(preds)
labels = np.concatenate(labels)
preds_real = cost_norm.unnormalize_labels(preds)
labels_real = cost_norm.unnormalize_labels(labels)


all_results = []
print("========== Validation Evaluation ==========")
for idx, (id_val, true_time, pred_time) in enumerate(zip(val_df['id'], labels_real, preds_real)):
    err = float(pred_time) - float(true_time)
    err_pct = err / float(true_time) if float(true_time) != 0 else 0
    err_pct_percent = round(err_pct * 100, 6)
    all_results.append({
        'id': id_val,
        'true': true_time,
        'pred': pred_time,
        'err': err,
        'err_pct': err_pct_percent
    })
    #print(f"{id_val}\t{float(true_time):.2f}\t{float(pred_time):.2f}\t{err:.2f}\t{err_pct_percent:.6f}%")


if "hybench5" in os.path.basename(data_file):
    output_filename = 'all_val_results_hybench5.csv'
elif "hybench10" in os.path.basename(data_file):
    output_filename = 'all_val_results_hybench10.csv'
elif "TPCH-5" in os.path.basename(data_file):
    output_filename = 'all_val_results_TPCH5.csv'
elif "TPCH-10" in os.path.basename(data_file):
    output_filename = 'all_val_results_TPCH10.csv'
else:
    output_filename = 'all_val_results.csv'

output_path = os.path.join(CURRENT_DIR, output_filename)
pd.DataFrame(all_results).to_csv(output_path, index=False)
print(f"Validation results saved to {output_path}")
