"""ABC Credit - two-wheeler loan approval chatbot (Streamlit).

Two modes:
  * Chat application: the assistant asks only the 5 inputs the model needs and returns Approve / Decline.
  * Batch scoring: upload an Excel/CSV test file; columns are matched to the model inputs, the model scores every
    row and the file is returned with the decision appended as the last column.
The decision always comes from the trained XGBoost model (predict.py); the LLM only reads answers and words replies.
Every interaction is logged to Google Sheets (see apps_script/Code.gs) with PII masked.
"""
import io
import os
import time
import uuid
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score

import chat_engine as ce
import column_mapper as cm
import predict
from llm import LLM
from pii import mask_pii
from sheets_logger import SheetsLogger

st.set_page_config(page_title='ABC Credit Loan Assistant', page_icon='🏍️', layout='centered')


def secret(key, default=''):
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:          # no secrets file at all
        pass
    return os.environ.get(key, default)


@st.cache_resource
def get_llm():
    return LLM(secret('GROQ_API_KEY'), secret('GROQ_MODEL') or None)


@st.cache_resource
def get_logger():
    return SheetsLogger(secret('GSHEET_WEBHOOK_URL'), secret('GSHEET_TOKEN'))


llm, logger = get_llm(), get_logger()
now_iso = lambda: datetime.now(timezone.utc).isoformat(timespec='seconds')


def new_chat():
    st.session_state.chat = {'id': uuid.uuid4().hex[:12], 'state': ce.new_state(), 'messages': []}
    st.session_state.chat['messages'].append({'role': 'assistant', 'content': ce.GREETING})
    logger.log('chat_log', [{'timestamp': now_iso(), 'session_id': st.session_state.chat['id'], 'turn': 0,
                             'action': 'greeting', 'assistant_message': ce.GREETING, 'llm': llm.name}])


if 'chat' not in st.session_state:
    new_chat()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title('ABC Credit')
    mode = st.radio('Mode', ['💬 Chat application', '📄 Batch scoring (Excel)'], label_visibility='collapsed')
    if st.button('🔄 New application', width='stretch'):
        new_chat()
        st.rerun()
    st.divider()
    st.caption('**How decisions are made**')
    st.caption(f'A trained XGBoost model scores each application; it is declined when the decline probability is '
               f'≥ {predict.THRESHOLD:.2f}. The language model only asks questions and explains the outcome.')
    st.caption('**Privacy:** no name or ID is asked for. Phone numbers, emails, Aadhaar and PAN typed by mistake '
               'are masked before anything is logged or sent to the language model.')
    st.divider()
    st.caption(f"Language model: {llm.name if llm.available else 'off (fixed wording)'}")
    st.caption(f"Google Sheets log: {logger.last_status if logger.enabled else 'off (no web-app URL)'}")

# ------------------------------------------------------------------ chat mode
def render_decision(d):
    (st.success if d['decision'] == 'APPROVED' else st.error)(f"**Loan {d['decision'].lower()}**")
    with st.expander('Model details (evaluator view)'):
        st.write({'decline_probability': round(d['probability'], 4), 'threshold': d['threshold'],
                  'make_used_by_model': d['make_used'], 'model_inputs': {k: v for k, v in d['features'].items()
                                                                         if not k.startswith('Make_')},
                  'data_warnings': d['warnings'] or 'none',
                  'model_ms': round(d['model_ms'], 2), 'decision_latency_ms': round(d['decision_ms'], 1),
                  'time_to_complete_s': round(d['session_seconds'], 1)})


def chat_mode():
    st.header('Two-wheeler loan: instant decision')
    chat = st.session_state.chat
    for m in chat['messages']:
        with st.chat_message(m['role']):
            if m.get('decision'):
                render_decision(m['decision'])
            st.markdown(m['content'])

    prompt = st.chat_input('Type your answer…')
    if not prompt:
        return
    chat['messages'].append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    state = chat['state']
    plan = ce.process(state, prompt, llm)
    msg = {'role': 'assistant'}
    with st.chat_message('assistant'):
        if plan['action'] == 'decision':
            msg['decision'] = {k: plan[k] for k in ('decision', 'probability', 'threshold', 'features', 'make_used',
                                                   'warnings', 'model_ms', 'decision_ms', 'session_seconds')}
            render_decision(msg['decision'])
        t = time.perf_counter()
        text = st.write_stream(ce.reply_stream(plan, state, llm))
        reply_ms = (time.perf_counter() - t) * 1000
    msg['content'] = text if isinstance(text, str) else ''.join(map(str, text))
    chat['messages'].append(msg)

    row = {'timestamp': now_iso(), 'session_id': chat['id'], 'turn': plan['turn'], 'action': plan['action'],
           'asked_field': plan.get('asked'), 'user_message_masked': plan['user_masked'],
           'assistant_message': mask_pii(msg['content']), 'extractor': plan.get('extractor'),
           'extracted': plan.get('extracted'), 'captured': plan.get('captured'), 'validation_errors': plan.get('errors'),
           'collected_values': plan.get('state'), 'extract_ms': round(plan.get('extract_ms', 0), 1),
           'reply_ms': round(reply_ms, 1), 'llm': llm.name if llm.available else 'none'}
    if plan['action'] == 'decision':
        f = plan['features']
        row.update(LTV=f['LTV'], Loan_Amount=f['Loan_Amount'], Age=f['Age'], PAST_LOANS_ACTIVE=f['PAST_LOANS_ACTIVE'],
                   make_used=plan['make_used'], decline_probability=round(plan['probability'], 6),
                   threshold=plan['threshold'], decision=plan['decision'], data_warnings=plan['warnings'],
                   model_ms=round(plan['model_ms'], 2), decision_latency_ms=round(plan['decision_ms'], 1),
                   time_to_complete_s=round(plan['session_seconds'], 1))
    logger.log('chat_log', [row])
    if plan['action'] == 'decision':
        st.rerun()     # redraw so the decision banner sits in the history


# ------------------------------------------------------------------ batch mode
@st.cache_data(show_spinner=False)
def read_upload(data: bytes, name: str, sheet):
    if name.lower().endswith('.csv'):
        return pd.read_csv(io.BytesIO(data))
    return pd.read_excel(io.BytesIO(data), sheet_name=sheet)


@st.cache_data(show_spinner=False)
def sheet_names(data: bytes):
    return pd.ExcelFile(io.BytesIO(data)).sheet_names


def to_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as w:
        df.to_excel(w, index=False, sheet_name='scored')
    return buf.getvalue()


def label_to_int(s: pd.Series) -> pd.Series:
    t = s.astype(str).str.strip().str.upper()
    return t.map(lambda v: 1 if v in ('DECLINED', 'DECLINE', '1', 'D', 'YES') else
                 (0 if v in ('APPROVED', 'APPROVE', '0', 'A', 'NO') else None))


def batch_mode():
    st.header('Batch scoring')
    st.write('Upload a test file (Excel or CSV). I will find the columns the model needs, score every row and add '
             'the decision as the last column.')
    up = st.file_uploader('Test dataset', type=['xlsx', 'xlsm', 'csv'])
    if not up:
        return
    data = up.getvalue()
    sheet = 0
    if not up.name.lower().endswith('.csv'):
        names = sheet_names(data)
        sheet = st.selectbox('Sheet', names) if len(names) > 1 else names[0]
    df = read_upload(data, up.name, sheet)
    st.caption(f'{len(df):,} rows × {df.shape[1]} columns')
    st.dataframe(df.head(5).astype(str), width='stretch')

    file_key = f'{up.name}|{len(data)}|{sheet}'
    if st.session_state.get('map_key') != file_key:
        with st.spinner('Matching columns…'):
            mapping, how, note = cm.auto_map(df, llm)
        st.session_state.update(map_key=file_key, mapping=mapping, map_how=how, map_note=note, batch_result=None)
    mapping, how = st.session_state.mapping, st.session_state.map_how
    if st.session_state.map_note:
        st.info(st.session_state.map_note)

    st.subheader('Column mapping')
    st.caption('Check the matches; change any that are wrong. Only these columns are sent to the model.')
    options = ['(none)'] + list(df.columns)
    chosen = {}
    cols = st.columns(2)
    for i, (f, (desc, _)) in enumerate(cm.FIELDS.items()):
        cur = mapping.get(f)
        label = f"{f} {'· ' + how[f] if f in how and cur else ''}"
        sel = cols[i % 2].selectbox(label, options, index=options.index(cur) if cur in options else 0,
                                    help=desc, key=f'map_{f}_{file_key}')
        chosen[f] = None if sel == '(none)' else sel
    issues = cm.problems(chosen)
    for p in issues:
        st.warning(p)

    if st.button('Run model', type='primary', disabled=any('No column' in p or 'same column' in p for p in issues)):
        t = time.perf_counter()
        raw = cm.build_model_input(df, chosen)
        scored, X = predict.score_detailed(raw)
        score_ms = (time.perf_counter() - t) * 1000
        out = df.copy()
        for c in ('Decline_Probability', 'Data_Warnings', 'Predicted_Decision'):
            out[c] = scored[c].values          # appended at the end; Predicted_Decision is the last column
        res = {'out': out, 'score_ms': score_ms, 'name': up.name, 'mapping': chosen}
        if chosen.get('Actual_Decision'):
            y = label_to_int(df[chosen['Actual_Decision']])
            keep = y.notna()
            if keep.any():
                yt, yp = y[keep].astype(int), (scored['Predicted_Decision'][keep] == 'DECLINED').astype(int)
                res['metrics'] = {'n': int(keep.sum()), 'cm': confusion_matrix(yt, yp, labels=[0, 1]),
                                  'acc': accuracy_score(yt, yp),
                                  'cls': {c: (precision_score(yt, yp, pos_label=c, zero_division=0),
                                              recall_score(yt, yp, pos_label=c, zero_division=0),
                                              f1_score(yt, yp, pos_label=c, zero_division=0)) for c in (0, 1)}}
        st.session_state.batch_result = res

        run_id = uuid.uuid4().hex[:12]
        n_dec = int((scored['Predicted_Decision'] == 'DECLINED').sum())
        run = {'timestamp': now_iso(), 'run_id': run_id, 'file_name': up.name, 'sheet': str(sheet), 'rows': len(df),
               'mapping': chosen, 'mapping_method': how, 'approved': len(df) - n_dec, 'declined': n_dec,
               'rows_with_warnings': int((scored['Data_Warnings'] != '').sum()), 'threshold': predict.THRESHOLD,
               'score_ms': round(score_ms, 1)}
        if 'metrics' in res:
            m = res['metrics']
            run.update(labelled_rows=m['n'], accuracy=round(m['acc'], 4),
                       declined_precision=round(m['cls'][1][0], 4), declined_recall=round(m['cls'][1][1], 4))
        logger.log('batch_runs', [run])
        used = predict.make_used(X)
        preds = [{'run_id': run_id, 'row_number': int(i) + 2, 'input_LTV': raw['LTV'].iat[j],
                  'input_Loan_Amount': raw['Loan_Amount'].iat[j], 'input_Age': raw['Age'].iat[j],
                  'input_PAST_LOANS_ACTIVE': raw['PAST_LOANS_ACTIVE'].iat[j], 'input_Make': raw['Make_Code'].iat[j],
                  'LTV': X['LTV'].iat[j], 'Loan_Amount': X['Loan_Amount'].iat[j], 'Age': X['Age'].iat[j],
                  'PAST_LOANS_ACTIVE': X['PAST_LOANS_ACTIVE'].iat[j], 'make_used': used.iat[j],
                  'decline_probability': scored['Decline_Probability'].iat[j],
                  'decision': scored['Predicted_Decision'].iat[j], 'data_warnings': scored['Data_Warnings'].iat[j]}
                 for j, i in enumerate(range(len(df)))]
        logger.log('batch_predictions', preds)

    res = st.session_state.get('batch_result')
    if not res:
        return
    out = res['out']
    n_dec = int((out['Predicted_Decision'] == 'DECLINED').sum())
    c1, c2, c3 = st.columns(3)
    c1.metric('Approved', f'{len(out) - n_dec:,}')
    c2.metric('Declined', f'{n_dec:,}')
    c3.metric('Scoring time', f"{res['score_ms']:.0f} ms")
    n_warn = int((out['Data_Warnings'] != '').sum())
    if n_warn:
        st.caption(f'{n_warn:,} rows had values fixed before scoring (see the Data_Warnings column).')
    if 'metrics' in res:
        m = res['metrics']
        st.subheader('Performance against the labelled outcome')
        (tn, fp), (fn, tp) = m['cm']
        st.table(pd.DataFrame([[tn, fp], [fn, tp]], index=['Actual APPROVED', 'Actual DECLINED'],
                              columns=['Predicted APPROVED', 'Predicted DECLINED']))
        st.table(pd.DataFrame({c: {'Precision': f'{v[0]:.2%}', 'Recall': f'{v[1]:.2%}', 'F1': f'{v[2]:.3f}'}
                               for c, v in (('APPROVED', m['cls'][0]), ('DECLINED', m['cls'][1]))}).T)
        st.caption(f"Accuracy {m['acc']:.2%} on {m['n']:,} labelled rows")
    st.dataframe(out.head(50).astype(str), width='stretch')
    stem = res['name'].rsplit('.', 1)[0]
    d1, d2 = st.columns(2)
    d1.download_button('⬇️ Download Excel', to_excel(out), f'{stem}_scored.xlsx', width='stretch',
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    d2.download_button('⬇️ Download CSV', out.to_csv(index=False).encode('utf-8'), f'{stem}_scored.csv',
                       mime='text/csv', width='stretch')


if mode.startswith('💬'):
    chat_mode()
else:
    batch_mode()
