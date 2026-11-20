"""Work out which uploaded column holds which model input.

1. Exact / synonym match on normalised header names (deterministic, instant).
2. For anything still unmapped, the LLM (if available) suggests a column from the headers and a few masked sample
   values. Suggestions are only accepted if the column really exists and is not already used.
The user can always override the mapping in the app before scoring.
"""
import json
import re
import pandas as pd
from pii import mask_pii

FIELDS = {   # canonical field -> (description, required?)
    'LTV': ('Loan-to-value ratio in percent', False),
    'Loan_Amount': ('Loan amount in rupees', True),
    'Age': ('Applicant age in years', True),
    'PAST_LOANS_ACTIVE': ('Whether the applicant has an active past loan (PAST_LOANS_ACTIVE / NO_PAST_LOANS, yes/no)', True),
    'Make_Code': ('Vehicle make or family, e.g. APACHE, JUPITER, RAIDER', False),
    'Vehicle_Price': ('Vehicle value / on-road price in rupees (used to compute LTV if there is no LTV column)', False),
    'Model_Description': ('Full vehicle model name, e.g. "TVS RAIDER DISC BSVI" (used for the make if Make_Code is missing)', False),
    'Actual_Decision': ('Known outcome APPROVED/DECLINED, only in labelled files (used to report accuracy)', False),
}
SYNONYMS = {
    'LTV': ['ltv', 'loantovalue', 'loantovalueratio', 'ltvratio', 'ltvpercent', 'ltvpct'],
    'Loan_Amount': ['loanamount', 'loanamt', 'loan', 'amount', 'financeamount', 'sanctionedamount', 'loanvalue'],
    'Age': ['age', 'applicantage', 'ageyears', 'customerage'],
    'PAST_LOANS_ACTIVE': ['pastloansactive', 'pastloans', 'activeloans', 'activeloan', 'existingloan', 'existingloans',
                          'priorloans', 'pastloan', 'previousloans'],
    'Make_Code': ['makecode', 'make', 'vehiclemake', 'brand', 'makename', 'manufacturer'],
    'Vehicle_Price': ['vehicleprice', 'onroadprice', 'assetcost', 'assetvalue', 'vehiclevalue', 'price', 'exshowroomprice'],
    'Model_Description': ['modeldescription', 'model', 'vehiclemodel', 'modelname', 'modeldesc'],
    'Actual_Decision': ['decision', 'actualdecision', 'loandecision', 'status', 'outcome', 'label', 'target'],
}


def _norm(h) -> str:
    return re.sub(r'[^a-z0-9]', '', str(h).lower())


def auto_map(df: pd.DataFrame, llm=None):
    """Return ({field: column or None}, {field: 'name match' | 'LLM' }, note)."""
    cols = list(df.columns)
    normed = {_norm(c): c for c in cols}
    mapping, how, used = dict.fromkeys(FIELDS), {}, set()
    for f, syns in SYNONYMS.items():
        for s in syns:
            c = normed.get(s)
            if c is not None and c not in used:
                mapping[f], how[f] = c, 'name match'
                used.add(c)
                break
    note = ''
    missing = [f for f in FIELDS if mapping[f] is None]
    free = [c for c in cols if c not in used]
    # ask the LLM only when something the model needs is still unmatched (optional extras alone are not worth a call)
    if problems(mapping) and free and llm is not None and llm.available:
        samples = {str(c): [mask_pii(str(v))[:40] for v in df[c].dropna().astype(str).unique()[:4]] for c in free}
        prompt = ('Map spreadsheet columns to loan-model fields. Return for each field the EXACT column name from '
                  'the list, or null if no column clearly holds that data. Use each column at most once.\n'
                  f'Fields: {json.dumps({f: FIELDS[f][0] for f in missing})}\n'
                  f'Columns with sample values: {json.dumps(samples, ensure_ascii=False)}')
        schema = {'type': 'object', 'additionalProperties': False,
                  'properties': {f: {'type': ['string', 'null']} for f in missing}, 'required': missing}
        try:
            sug = llm.json([{'role': 'user', 'content': prompt}], schema, name='mapping')
            for f in missing:
                c = sug.get(f)
                if c in free and c not in used:
                    mapping[f], how[f] = c, 'LLM'
                    used.add(c)
        except Exception as e:
            note = f'LLM column matching unavailable ({str(e)[:80]}); used name matching only.'
    return mapping, how, note


def problems(mapping: dict) -> list[str]:
    """What is still missing before the model can run."""
    out = []
    for f in ('Loan_Amount', 'Age', 'PAST_LOANS_ACTIVE'):
        if not mapping.get(f):
            out.append(f'No column selected for {f}.')
    if not mapping.get('LTV') and not mapping.get('Vehicle_Price'):
        out.append('Select an LTV column, or a Vehicle_Price column so LTV can be computed.')
    if not mapping.get('Make_Code') and not mapping.get('Model_Description'):
        out.append('Select a Make_Code or Model_Description column (otherwise every vehicle is scored as OTHER).')
    chosen = [c for c in mapping.values() if c]
    if len(chosen) != len(set(chosen)):
        out.append('The same column is selected for two fields.')
    return out


def build_model_input(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Raw-format frame with exactly the columns predict.score expects."""
    g = lambda f: df[mapping[f]] if mapping.get(f) else pd.Series([None] * len(df), index=df.index)
    raw = pd.DataFrame(index=df.index)
    ltv = pd.to_numeric(g('LTV'), errors='coerce')
    if mapping.get('Vehicle_Price'):
        price = pd.to_numeric(g('Vehicle_Price'), errors='coerce')
        loan = pd.to_numeric(g('Loan_Amount'), errors='coerce')
        computed = (loan / price.where(price > 0) * 100)
        ltv = ltv.fillna(computed) if mapping.get('LTV') else computed
    raw['LTV'] = ltv if mapping.get('LTV') or mapping.get('Vehicle_Price') else g('LTV')
    raw['Loan_Amount'] = g('Loan_Amount')
    raw['Age'] = g('Age')
    raw['PAST_LOANS_ACTIVE'] = g('PAST_LOANS_ACTIVE')
    make = g('Make_Code').astype('string').str.strip()
    if mapping.get('Model_Description'):
        make = make.where(make.notna() & (make != ''), g('Model_Description').astype('string'))
    raw['Make_Code'] = make.astype(object).where(make.notna(), None)
    return raw
