

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from model.dataset import PlanTreeDataset
from model.database_util import collator, get_job_table_sample
import os
import time
import torch
from scipy.stats import pearsonr


def create_mock_htap_data(batch_size, n_nodes, device='cpu'):

    htap_data = {

        'storage_type': torch.randint(0, 2, (batch_size, n_nodes), device=device),
        'insert_cnt': torch.rand(batch_size, n_nodes, device=device) * 0.1,
        'update_cnt': torch.rand(batch_size, n_nodes, device=device) * 0.2,
        'delete_cnt': torch.rand(batch_size, n_nodes, device=device) * 0.05,
        'query_type': torch.full((batch_size, n_nodes), 3, device=device, dtype=torch.long),
        'node_operator': torch.randint(0, 20, (batch_size, n_nodes), device=device)
    }
    return htap_data


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def print_qerror(preds_unnorm, labels_unnorm, prints=False):
    qerror = []
    for i in range(len(preds_unnorm)):
        if preds_unnorm[i] > float(labels_unnorm[i]):
            qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
        else:
            qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))

    e_50, e_90 = np.median(qerror), np.percentile(qerror, 90)    
    e_mean = np.mean(qerror)

    if prints:
        print("Median: {}".format(e_50))
        print("Mean: {}".format(e_mean))

    res = {
        'q_median': e_50,
        'q_90': e_90,
        'q_mean': e_mean,
    }

    return res


def get_corr(ps, ls):  # unnormalised
    ps = np.array(ps)
    ls = np.array(ls)
    corr, _ = pearsonr(np.log(ps), np.log(ls))
    
    return corr


def evaluate_htapformer(model, ds, bs, norm, device, prints=False, use_htap=True):
    """
    HtapFormer 评估函数
    
    Args:
        model: HtapFormer 模型
        ds: 数据集
        bs: batch size
        norm: Normalizer
        device: 设备
        prints: 是否打印结果
        use_htap: 是否使用 HTAP 特征（False 时等同于 QueryFormer）
    
    Returns:
        scores: 评估指标字典
    """
    model.eval()
    cost_predss = np.empty(0)

    with torch.no_grad():
        for i in range(0, len(ds), bs):
            batch, batch_labels = collator(list(zip(*[ds[j] for j in range(i, min(i+bs, len(ds)))])))
            batch = batch.to(device)

            if use_htap:
                batch_size = batch.x.size(0)
                n_nodes = batch.x.size(1)
                htap_data = create_mock_htap_data(batch_size, n_nodes, device)
                cost_preds, _ = model(batch, htap_data)
            else:
                cost_preds, _ = model(batch, htap_data=None)
            
            cost_preds = cost_preds.squeeze()
            cost_predss = np.append(cost_predss, cost_preds.cpu().detach().numpy())
    
    scores = print_qerror(norm.unnormalize_labels(cost_predss), ds.costs, prints)
    corr = get_corr(norm.unnormalize_labels(cost_predss), ds.costs)
    
    if prints:
        print("Corr: {}".format(corr))
    
    scores['corr'] = corr
    return scores


def eval_workload_htapformer(workload, methods, use_htap=True):

    get_table_sample = methods['get_sample']

    workload_file_name = './data/imdb/workloads/' + workload
    table_sample = get_table_sample(workload_file_name)
    plan_df = pd.read_csv('./data/imdb/{}_plan.csv'.format(workload))
    workload_csv = pd.read_csv('./data/imdb/workloads/{}.csv'.format(workload), sep='#', header=None)
    workload_csv.columns = ['table', 'join', 'predicate', 'cardinality']
    
    ds = PlanTreeDataset(
        plan_df, workload_csv,
        methods['encoding'], methods['hist_file'], 
        methods['cost_norm'], methods['cost_norm'], 
        'cost', table_sample
    )

    eval_score = evaluate_htapformer(
        methods['model'], ds, methods['bs'], 
        methods['cost_norm'], methods['device'], 
        prints=True, use_htap=use_htap
    )
    
    return eval_score, ds


def train_htapformer(model, train_ds, val_ds, crit, norm, args, use_htap=True):
    import torch.optim as optim
    from torch.utils.data import DataLoader
    
    # 创建优化器
    opt = optim.Adam(model.parameters(), lr=args.lr)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=1, gamma=args.sch_decay)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.bs, 
        shuffle=True,
        collate_fn=lambda x: collator(list(zip(*x)))
    )
    
    best_val_loss = float('inf')
    best_epoch = 0
    best_path = None
    
    print(f"{'='*60}")
    print(f"Training HtapFormer")
    print(f"  Use HTAP features: {use_htap}")
    print(f"  HTAP lambda: {getattr(args, 'htap_lambda', 'N/A')}")
    print(f"  Training samples: {len(train_ds)}")
    print(f"  Validation samples: {len(val_ds)}")
    print(f"  Batch size: {args.bs}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"{'='*60}\n")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        start_time = time.time()
        
        for batch_idx, (batch, batch_labels) in enumerate(train_loader):
            batch = batch.to(args.device)
            
            l, r = zip(*batch_labels)
            batch_cost_label = torch.FloatTensor(l).to(args.device)
            
            if use_htap:
                batch_size = batch.x.size(0)
                n_nodes = batch.x.size(1)
                htap_data = create_mock_htap_data(batch_size, n_nodes, args.device)
                cost_preds, _ = model(batch, htap_data)
            else:
                cost_preds, _ = model(batch, htap_data=None)
            
            cost_preds = cost_preds.squeeze()
            loss = crit(cost_preds, batch_cost_label)
            
            opt.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_size)
            
            opt.step()
            
            del batch
            del batch_labels
            if args.device.startswith('cuda'):
                torch.cuda.empty_cache()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss = epoch_loss / num_batches
        epoch_time = time.time() - start_time
        
        val_score = evaluate_htapformer(
            model, val_ds, args.bs, norm, 
            args.device, prints=False, use_htap=use_htap
        )
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Q-Median: {val_score['q_median']:.2f}")
        print(f"  Val Q-Mean: {val_score['q_mean']:.2f}")
        print(f"  Val Correlation: {val_score['corr']:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Epoch Time: {epoch_time:.2f}s\n")
        
        if val_score['q_median'] < best_val_loss:
            best_val_loss = val_score['q_median']
            best_epoch = epoch + 1
            
            model_hash = hash(time.time())
            best_path = os.path.join(args.newpath, f'{model_hash}.pt')
            torch.save(model.state_dict(), best_path)
            
            print(f"  ✅ Best model saved! Q-Median: {best_val_loss:.2f} at epoch {best_epoch}")
            print(f"     Path: {best_path}\n")
        
        logging(args, epoch + 1, val_score, filename='log.txt', save_model=False)
    
    print(f"\n{'='*60}")
    print(f"Training Completed!")
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Best Val Q-Median: {best_val_loss:.2f}")
    print(f"  Best Model: {best_path}")
    print(f"{'='*60}\n")
    
    return model, best_path


def logging(args, epoch, test_scores, filename='log.txt', save_model=True, model=None):

    logfile = os.path.join(args.newpath, filename)
    
    res = {
        'epoch': epoch,
        **test_scores
    }
    
    if os.path.exists(logfile):
        df = pd.read_csv(logfile)
    else:
        df = pd.DataFrame()
    
    res_df = pd.DataFrame([res])
    df = pd.concat([df, res_df], ignore_index=True)
                    
    df.to_csv(logfile, index=False)
    
    best_model_path = None
    if save_model and model is not None:
        model_hash = hash(time.time())
        best_model_path = os.path.join(args.newpath, f'{model_hash}.pt')
        torch.save(model.state_dict(), best_model_path)
    
    return best_model_path

