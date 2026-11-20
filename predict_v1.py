"""Score loan applications with the saved XGBoost model.

Usage:
    python predict.py <input.xlsx|input.csv> [sheet_name] [output.csv]

The input needs the raw columns LTV, Loan_Amount, Age, PAST_LOANS_ACTIVE (PAST_LOANS_ACTIVE / NO_PAST_LOANS)
and Make_Code (e.g. APACHE, JUPITER). Other columns are kept in the output untouched.
Output adds Decline_Probability and Predicted_Decision (DECLINED if probability >= saved threshold).
"""
import json
import sys
from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
meta = json.load(open(HERE / 'xgb_model_meta_v1.json'))
model = XGBClassifier(); model.load_model(HERE / 'xgb_model_v1.json')
FEATURES, THRESHOLD = meta['features'], meta['threshold']


def encode(raw: pd.DataFrame) -> pd.DataFrame:
    missing = {'LTV', 'Loan_Amount', 'Age', 'PAST_LOANS_ACTIVE', 'Make_Code'} - set(raw.columns)
    if missing:
        raise ValueError(f'missing columns: {sorted(missing)}')
    X = raw[['LTV', 'Loan_Amount', 'Age']].astype(float).copy()
    X['PAST_LOANS_ACTIVE'] = (raw['PAST_LOANS_ACTIVE'].astype(str).str.strip().str.upper() == 'PAST_LOANS_ACTIVE').astype(int)
    make = 'Make_' + raw['Make_Code'].astype(str).str.strip().str.upper().str.replace(' ', '_')
    known = [f for f in FEATURES if f.startswith('Make_')]
    unknown = sorted(set(make) - set(known))
    if unknown:
        print(f'WARNING: makes not seen in training (scored with all Make columns = 0): {unknown}')
    for f in known:
        X[f] = (make == f).astype(int)
    return X[FEATURES]


def score(raw: pd.DataFrame) -> pd.DataFrame:
    p = model.predict_proba(encode(raw))[:, 1]
    out = raw.copy()
    out['Decline_Probability'] = p.round(4)
    out['Predicted_Decision'] = ['DECLINED' if v >= THRESHOLD else 'APPROVED' for v in p]
    return out


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1])
    sheet = sys.argv[2] if len(sys.argv) > 2 else 0
    raw = pd.read_csv(src) if src.suffix.lower() == '.csv' else pd.read_excel(src, sheet_name=sheet)
    out = score(raw)
    dst = Path(sys.argv[3]) if len(sys.argv) > 3 else src.with_name(src.stem + '_scored.csv')
    out.to_csv(dst, index=False)
    print(f'threshold {THRESHOLD:.4f} | scored {len(out)} rows -> {dst}')
    print(out['Predicted_Decision'].value_counts().to_string())
