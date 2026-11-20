"""Train the loan-decision XGBoost model (v2) from the training_data_cleaned sheet.

Decision: 1 = DECLINED, 0 = APPROVED. Steps:
  1. 80/20 stratified train/test split (seed 42, same rows as v1); encoding settings learned from train only.
  2. Two candidates, with and without monotonic constraints (risk rises with LTV, falls with Age).
     Each: early stopping on a 15% validation slice of train, then 5-fold out-of-fold (OOF) scores on train.
     Keep constraints unless OOF PR-AUC drops by more than 0.002.
  3. Decision threshold fixed at 0.53; the F2-optimal value on the OOF scores is saved for reference only.
     Test set is used only at the end.
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (confusion_matrix, precision_recall_curve, precision_score, recall_score,
                             accuracy_score, roc_auc_score, average_precision_score)
from xgboost import XGBClassifier
from pipeline import fit_spec, encode

SEED = 42
raw = pd.read_excel('training_data.xlsx', sheet_name='training_data_cleaned')
y = (raw['Decision'] == 'DECLINED').astype(int)
idx_tr, idx_te = train_test_split(raw.index, test_size=0.20, stratify=y, random_state=SEED)
spec = fit_spec(raw.loc[idx_tr])
X, enc_warn = encode(raw, spec)
assert not any(enc_warn[i] for i in raw.index.get_indexer(idx_tr)), 'training rows should need no fixes'
X_tr, y_tr, X_te, y_te = X.loc[idx_tr], y.loc[idx_tr], X.loc[idx_te], y.loc[idx_te]
X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=SEED)
spw = float((y_fit == 0).sum() / (y_fit == 1).sum())
print('features:', spec['features']); print('grouped as OTHER:', spec['grouped_makes'], '| scale_pos_weight', round(spw, 2))

BASE = dict(learning_rate=0.03, max_depth=5, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, scale_pos_weight=spw, objective='binary:logistic', tree_method='hist',
            random_state=SEED, n_jobs=-1)
CANDIDATES = {'unconstrained': {}, 'monotonic': {'monotone_constraints': {'LTV': 1, 'Age': -1}}}

def fit_candidate(extra):
    es = XGBClassifier(n_estimators=2000, eval_metric='aucpr', early_stopping_rounds=100, **BASE, **extra)
    es.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    n_trees = es.best_iteration + 1
    oof = np.zeros(len(y_tr))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X_tr, y_tr):
        m = XGBClassifier(n_estimators=n_trees, **BASE, **extra).fit(X_tr.iloc[a], y_tr.iloc[a])
        oof[b] = m.predict_proba(X_tr.iloc[b])[:, 1]
    return es, n_trees, oof

res = {}
for name, extra in CANDIDATES.items():
    es, n_trees, oof = fit_candidate(extra)
    res[name] = (es, n_trees, oof, average_precision_score(y_tr, oof), roc_auc_score(y_tr, oof))
    print(f'{name:14s} trees {n_trees:4d} | OOF PR-AUC {res[name][3]:.4f}  ROC-AUC {res[name][4]:.4f}')
choice = 'monotonic' if res['monotonic'][3] >= res['unconstrained'][3] - 0.002 else 'unconstrained'
model, n_trees, oof, _, _ = res[choice]
print('chosen:', choice)

P, R, T = precision_recall_curve(y_tr, oof)
f2 = 5 * P[:-1] * R[:-1] / (4 * P[:-1] + R[:-1] + 1e-12)
thr_f2 = float(T[f2.argmax()])
print(f'F2-optimal threshold (OOF, reference only) {thr_f2:.4f}: P {P[f2.argmax()]:.4f} R {R[f2.argmax()]:.4f}')
thr = 0.53  # decision threshold fixed at 0.53 by choice

p = model.predict_proba(X_te)[:, 1]
pred = (p >= thr).astype(int)
cm = confusion_matrix(y_te, pred, labels=[0, 1])
print(f'\nTEST (n={len(y_te)}, declined={int(y_te.sum())}) threshold {thr:.4f}')
print('CM [[TN, FP], [FN, TP]]:', cm.tolist())
for c, nm in ((0, 'APPROVED'), (1, 'DECLINED')):
    print(f'  {nm}: precision {precision_score(y_te, pred, pos_label=c):.4f}  recall {recall_score(y_te, pred, pos_label=c):.4f}')
print(f'  accuracy {accuracy_score(y_te, pred):.4f} | ROC-AUC {roc_auc_score(y_te, p):.4f} | PR-AUC {average_precision_score(y_te, p):.4f}')

model.save_model('xgb_model.json')
json.dump({'version': 2, 'threshold': thr, 'threshold_f2': thr_f2,
           'threshold_basis': 'fixed at 0.53 by user choice (F2-optimal value kept as threshold_f2 for reference)',
           'label': {'1': 'DECLINED', '0': 'APPROVED'}, 'model_variant': choice,
           'monotone_constraints': CANDIDATES[choice].get('monotone_constraints', {}),
           'n_trees': int(n_trees), 'scale_pos_weight': spw, 'seed': SEED, 'spec': spec},
          open('xgb_model_meta.json', 'w'), indent=2)
print('saved xgb_model.json + xgb_model_meta.json')
