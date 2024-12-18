import math

import torch
import numpy as np
from scipy.stats import stats

from train_utils import test_xk

"""
This code is from Hwang et al., CVPR 2024. https://github.com/y0ngjaenius/CVPR2024_FLOWERFormer
"""


def train_1cell(model, loader, optimizer, device, cfg):
    model.train()
    total_loss = 0
    tot = 0
    for batch, _ in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        sample_scores, y = model(batch)
        sample_scores = sample_scores.squeeze()
        if sample_scores.ndim == 0:
            continue
        if cfg.dag.compare.do_limit:
            n_max_pairs = int(cfg.dag.compare.max_compare_ratio * len(batch))
        else:
            n_max_pairs = math.inf
        y = y.cpu().detach().numpy()
        acc_diff = y[:, None] - y
        acc_abs_diff_matrix = np.triu(np.abs(acc_diff), 1)
        ex_thresh_inds = np.where(acc_abs_diff_matrix > 0.0)
        ex_thresh_num = len(ex_thresh_inds[0])

        if ex_thresh_num > n_max_pairs:
            keep_inds = np.random.choice(np.arange(ex_thresh_num), n_max_pairs, replace=False)
            ex_thresh_inds = (ex_thresh_inds[0][keep_inds], ex_thresh_inds[1][keep_inds])

        better_labels = (acc_diff > 0)[ex_thresh_inds]
        n_diff_pairs = len(better_labels)

        s_1 = sample_scores[ex_thresh_inds[1]]
        s_2 = sample_scores[ex_thresh_inds[0]]

        better_pm = 2 * s_1.new(np.array(better_labels, dtype=np.float32)) - 1
        zero_, margin = s_1.new([0.0]), s_1.new([cfg.dag.compare.margin])

        loss = torch.mean(torch.max(zero_, margin - better_pm * (s_2 - s_1)))

        loss.backward()
        if cfg.optim.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.clip_grad_norm_value)
        optimizer.step()
        total_loss += float(loss) * n_diff_pairs
        tot += n_diff_pairs
    train_loss = total_loss / tot
    return train_loss


@torch.no_grad()
def eval_1cell(model, loader, device):
    model.eval()

    all_scores = []
    true_accs = []

    for batch, _ in loader:
        batch = batch.to(device)
        output, y = model(batch)

        all_scores.extend(output.squeeze().cpu().tolist())
        true_accs.extend(y.squeeze().cpu().tolist())

    kt = stats.kendalltau(true_accs, all_scores).correlation
    sp = stats.spearmanr(true_accs, all_scores).correlation
    pak = test_xk(true_accs, all_scores)
    return {"kt": kt, "sp": sp, "pak": pak}

#train_dict = {"1cell": (train_1cell, eval_1cell), "2cell": (train_2cell, eval_2cell)}
train_dict = {"1cell": (train_1cell, eval_1cell), "2cell": (None, None)}
