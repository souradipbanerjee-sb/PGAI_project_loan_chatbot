"""Score loan applications with the saved XGBoost model (v2).

Usage:
    python predict.py <input.xlsx|input.csv> [sheet_name] [output.csv]

Needs the raw columns LTV, Loan_Amount, Age, PAST_LOANS_ACTIVE and Make_Code (other columns, e.g. Net_salary,
are passed through untouched: the model does not use them). Input is validated before scoring:
  - numbers outside the training range are clipped to it; impossible, non-numeric or missing values -> training median
  - Make_Code is matched case/space-insensitively, then by known make name inside the text
    (e.g. "TVS APACHE RTR" -> APACHE); anything else -> OTHER (a group the model was trained on)
  - PAST_LOANS_ACTIVE accepts PAST_LOANS_ACTIVE/NO_PAST_LOANS, yes/no, y/n, 1/0, true/false
Every fix is listed per row in the Data_Warnings column.
"""
import json
import sys
from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier
from pipeline import encode

HERE = Path(__file__).resolve().parent
meta = json.load(open(HERE / 'xgb_model_meta.json'))
model = XGBClassifier(); model.load_model(HERE / 'xgb_model.json')
SPEC, THRESHOLD = meta['spec'], meta['threshold']


def score_detailed(raw: pd.DataFrame):
    """Return (raw rows + Decline_Probability, Data_Warnings, Predicted_Decision as the last column,
    the encoded feature matrix actually fed to the model)."""
    X, warns = encode(raw, SPEC)
    p = model.predict_proba(X)[:, 1]
    out = raw.copy()
    out['Decline_Probability'] = p.round(4)
    out['Data_Warnings'] = warns
    out['Predicted_Decision'] = ['DECLINED' if v >= THRESHOLD else 'APPROVED' for v in p]
    return out, X


def score(raw: pd.DataFrame) -> pd.DataFrame:
    return score_detailed(raw)[0]


def make_used(X: pd.DataFrame) -> pd.Series:
    """Which make group the model actually used for each row (e.g. APACHE, OTHER)."""
    cols = [c for c in X.columns if c.startswith('Make_')]
    return X[cols].idxmax(axis=1).str[len('Make_'):]


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
    print(f'rows with data warnings: {(out["Data_Warnings"] != "").sum()}')
