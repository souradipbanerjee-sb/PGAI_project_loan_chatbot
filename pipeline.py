"""Shared preprocessing for training and scoring: input validation, make mapping and one-hot encoding.

The same `encode` is used when training and when scoring, so new data is always prepared exactly like the
training data. All learned settings (kept makes, numeric ranges, defaults) live in a JSON-serialisable `spec`.
"""
import re
import numpy as np
import pandas as pd

NUM = ['LTV', 'Loan_Amount', 'Age']
MIN_MAKE_SHARE = 0.01          # makes below 1% of training rows are grouped as OTHER
BRAND_WORDS = {'TVS'}          # manufacturer name: too generic to match inside a longer make string
# Names applicants use for the training makes (from Model_Variant / Model_Description in the training data).
# Checked before plain make-name matching, most specific first (e.g. "SCOOTY ZEST" is the ZEST make, not SCOOTY).
MAKE_ALIASES = [
    (r'SCOOTY\s*ZEST', 'ZEST'), (r'SCOOTY\s*PEP', 'SCOOTY'), (r'STAR\s*CITY', 'CITY PLUS'),
    (r'XL\s*-?\s*100', 'MOPEDS'), (r'XL', 'MOPEDS'), (r'MOPEDS?', 'MOPEDS'),
    (r'E[\s-]*BIKE', 'TVS'), (r'IQUBE', 'TVS'), (r'ELECTRIC', 'TVS'), (r'EV', 'TVS'),
    (r'N\s*TORQ', 'NTORQ'),
]
PAST_YES = {'PAST_LOANS_ACTIVE', 'YES', 'Y', '1', '1.0', 'TRUE', 'ACTIVE'}
PAST_NO = {'NO_PAST_LOANS', 'NO', 'N', '0', '0.0', 'FALSE', 'NONE'}
# values that cannot be real: treated as missing (then median-filled), not just clipped
IMPOSSIBLE = {'LTV': lambda v: v < 0, 'Loan_Amount': lambda v: v <= 0, 'Age': lambda v: (v <= 0) | (v > 120)}


def norm_make(v) -> str:
    return '' if pd.isna(v) else re.sub(r'\s+', ' ', str(v).strip().upper())


def make_col(m: str) -> str:
    return 'Make_' + m.replace(' ', '_')


def fit_spec(raw: pd.DataFrame) -> dict:
    """Learn encoding settings from TRAINING rows only."""
    makes = raw['Make_Code'].map(norm_make)
    share = makes.value_counts(normalize=True)
    kept = sorted(share[share >= MIN_MAKE_SHARE].index)
    past_active_share = (raw['PAST_LOANS_ACTIVE'].astype(str).str.strip().str.upper() == 'PAST_LOANS_ACTIVE').mean()
    return {
        'kept_makes': kept,
        'grouped_makes': sorted(set(share.index) - set(kept)),
        'numeric': {c: {'min': float(raw[c].min()), 'max': float(raw[c].max()), 'median': float(raw[c].median())}
                    for c in NUM},
        'past_default': int(past_active_share >= 0.5),
        'features': NUM + ['PAST_LOANS_ACTIVE'] + [make_col(m) for m in kept] + ['Make_OTHER'],
    }


def _word(pattern: str, text: str) -> bool:
    return re.search(r'(?<![A-Z0-9])' + pattern + r'(?![A-Z0-9])', text) is not None


def map_make(m: str, spec: dict):
    """Return (make used by the model, warning or None)."""
    kept, grouped = spec['kept_makes'], spec['grouped_makes']
    if m in kept:
        return m, None
    if m in grouped:
        return 'OTHER', None                     # rare in training: OTHER by design
    if not m:
        return 'OTHER', 'Make_Code missing -> OTHER'
    for alias, k in MAKE_ALIASES:                # model / category names people use for the training makes
        if _word(alias, m) and k in kept + grouped:
            target = k if k in kept else 'OTHER'
            return target, f'Make_Code "{m}" matched to {k}' + (' (grouped as OTHER)' if target == 'OTHER' else '')
    for k in sorted([x for x in kept + grouped if x not in BRAND_WORDS], key=len, reverse=True):
        if re.search(r'(?<![A-Z0-9])' + re.escape(k) + r'(?![A-Z0-9])', m):
            target = k if k in kept else 'OTHER'
            return target, f'Make_Code "{m}" matched to {k}' + (' (grouped as OTHER)' if target == 'OTHER' else '')
    return 'OTHER', f'Make_Code "{m}" not seen in training -> OTHER'


def encode(raw: pd.DataFrame, spec: dict):
    """Validate and encode raw rows. Returns (X with spec['features'] columns, list of warning strings per row)."""
    missing_cols = {'LTV', 'Loan_Amount', 'Age', 'PAST_LOANS_ACTIVE', 'Make_Code'} - set(raw.columns)
    if missing_cols:
        raise ValueError(f'input is missing columns: {sorted(missing_cols)}')
    n = len(raw)
    warns = [[] for _ in range(n)]
    X = pd.DataFrame(index=raw.index)

    for c in NUM:
        rng = spec['numeric'][c]
        v = pd.to_numeric(raw[c], errors='coerce').astype(float)
        bad = IMPOSSIBLE[c](v.fillna(1)) & v.notna()
        nonnum = v.isna() & raw[c].notna() & (raw[c].astype(str).str.strip() != '')
        for i in np.flatnonzero(bad.values):
            warns[i].append(f'{c}={v.iat[i]:g} impossible -> median {rng["median"]:g}')
        for i in np.flatnonzero(nonnum.values):
            warns[i].append(f'{c}="{raw[c].iat[i]}" not a number -> median {rng["median"]:g}')
        v = v.mask(bad)
        for i in np.flatnonzero((v.isna() & ~bad & ~nonnum).values):
            warns[i].append(f'{c} missing -> median {rng["median"]:g}')
        v = v.fillna(rng['median'])
        lo, hi = v < rng['min'], v > rng['max']
        for i in np.flatnonzero(lo.values):
            warns[i].append(f'{c}={v.iat[i]:g} below training min {rng["min"]:g} -> clipped')
        for i in np.flatnonzero(hi.values):
            warns[i].append(f'{c}={v.iat[i]:g} above training max {rng["max"]:g} -> clipped')
        X[c] = v.clip(rng['min'], rng['max'])

    past = raw['PAST_LOANS_ACTIVE'].map(lambda s: '' if pd.isna(s) else str(s).strip().upper())
    vals = []
    for i, s in enumerate(past):
        if s in PAST_YES:
            vals.append(1)
        elif s in PAST_NO:
            vals.append(0)
        else:
            vals.append(spec['past_default'])
            warns[i].append(f'PAST_LOANS_ACTIVE "{s}" not recognised -> {spec["past_default"]}' if s
                            else f'PAST_LOANS_ACTIVE missing -> {spec["past_default"]}')
    X['PAST_LOANS_ACTIVE'] = vals

    mapped = []
    for i, m in enumerate(raw['Make_Code'].map(norm_make)):
        t, w = map_make(m, spec)
        mapped.append(t)
        if w:
            warns[i].append(w)
    mapped = pd.Series(mapped, index=raw.index)
    for m in spec['kept_makes']:
        X[make_col(m)] = (mapped == m).astype(int)
    X['Make_OTHER'] = (mapped == 'OTHER').astype(int)
    return X[spec['features']], ['; '.join(w) for w in warns]
