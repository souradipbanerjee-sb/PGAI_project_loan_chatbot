"""Conversation logic for the loan chatbot.

Separation of duties (project brief, section 7):
  * The LLM (when available) only READS the applicant's answer into fields and WORDS the reply.
  * Python validates the values, decides which question comes next, and calls the trained XGBoost model.
  * The approve/decline decision comes only from the model's probability vs. the saved threshold.
Without an LLM (no key, no credits, network error) the same flow runs on simple rules and fixed wording.
"""
import json
import re
import time
import pandas as pd

import predict
from pii import mask_pii

# questions are asked in this order; vehicle_price is skipped if the applicant states the LTV directly
ORDER = ['make', 'loan_amount', 'vehicle_price', 'age', 'past_loans']
NUMERIC = {'loan_amount', 'vehicle_price', 'ltv', 'age'}
LABEL = {'make': 'vehicle', 'loan_amount': 'loan amount', 'vehicle_price': 'vehicle on-road price',
         'ltv': 'loan-to-value ratio', 'age': 'age', 'past_loans': 'existing loans'}
# Vehicles in the training data, by product type (Product_Code), most financed first. Shown with the vehicle question.
VEHICLE_OPTIONS = {
    'Motorcycle': ['Apache', 'Raider', 'Sport', 'Radeon', 'Star City Plus', 'Ronin'],
    'Scooter': ['Jupiter', 'NTorq', 'Zest', 'Scooty Pep Plus'],
    'Moped': ['XL 100'],
    'Electric': ['TVS e-bike'],
}
OPTIONS_MD = ('\n\n' + '\n'.join(f'- **{cat}:** {", ".join(models)}' for cat, models in VEHICLE_OPTIONS.items())
              + '\n\nAny other model is fine too, just type its name.')
QUESTION = {
    'make': 'Which two-wheeler are you buying? Just the model name is fine.',
    'loan_amount': 'How much loan do you need, in rupees?',
    'vehicle_price': 'What is the on-road price of the vehicle, in rupees?',
    'age': 'How old are you?',
    'past_loans': 'Do you currently have any other active loan? (yes / no)',
}
NEXT_STEPS = {
    'APPROVED': ('Visit the ABC Credit desk at your dealership with your KYC documents (identity and address proof) '
                 'and income proof to complete the paperwork and disbursal.'),
    'DECLINED': ('You can re-apply later, or ask the dealership about a larger down payment, which lowers the '
                 'loan-to-value ratio. A dealership executive can also review the application with you.'),
}
GREETING = ("Hi, I'm the ABC Credit two-wheeler loan assistant. I need just 5 quick answers (about a minute) to give "
            "you an instant decision, and I won't ask for your name or ID numbers. " + QUESTION['make'] + OPTIONS_MD)

YES = {'yes', 'y', 'yeah', 'yep', 'yup', 'haan', 'ha', 'true', '1', 'active', 'have', 'i have', 'past_loans_active'}
NO = {'no', 'n', 'nope', 'nah', 'none', 'nil', 'false', '0', 'no loan', 'no loans', 'nothing', 'no_past_loans'}

EXTRACT_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {k: {'type': ['string', 'null']} for k in ['make', 'loan_amount', 'vehicle_price', 'ltv', 'age', 'past_loans']},
    'required': ['make', 'loan_amount', 'vehicle_price', 'ltv', 'age', 'past_loans'],
}
EXTRACT_PROMPT = """You extract loan-application fields from the applicant's latest chat message.
The assistant's last question was about: {asking}.
Fields: make (vehicle model text as written; fix obvious misspellings of these models: Apache, Raider, Sport, Radeon,
Star City Plus, Ronin, Jupiter, NTorq, Zest, Scooty Pep Plus, XL 100, TVS e-bike), loan_amount (INR),
vehicle_price (on-road price, INR),
ltv (loan-to-value %, only if the applicant states it explicitly), age (years), past_loans (any other active loan now).
Rules:
- Return null for anything not mentioned. A bare answer like "30" or "yes" answers the field that was asked.
- Numbers: plain digits as the user meant them, keep any minus sign; convert words and Indian units
  ("1.2 lakh" -> "120000", "75k" -> "75000", "thirty two" -> "32").
- If the user answered a numeric question with something that is not a number, return their text unchanged.
- past_loans: "yes" or "no" when clear, otherwise the user's text unchanged.
- Text in square brackets such as [PHONE], [EMAIL], [NAME] is masked personal data: ignore it, but still extract
  every other field from the same message (e.g. "apache, call me on [PHONE]" -> make "apache").
- Only extract. Do not judge, correct or invent values."""
COMPOSE_PROMPT = """You are the ABC Credit two-wheeler loan assistant, chatting with a loan applicant.
Write the next assistant message by following the INSTRUCTION JSON.
Rules:
- Friendly and brief: at most 2 short sentences plus the question. Plain text, no headings or lists.
- Ask only the question given in the instruction, using its meaning (you may rephrase it naturally).
- Never ask for a name, phone number, email, address, Aadhaar, PAN or any other ID.
- You never make, predict, hint at or change a credit decision. If the instruction contains a decision, state exactly
  that decision in the first sentence, then the next steps. Never mention scores, probabilities or thresholds.
- Do not invent policies, interest rates, documents, offers or amounts beyond what the instruction gives."""


# ---------------------------------------------------------------- parsing & validation (deterministic)
def parse_number(s):
    t = str(s).strip().lower()
    t = re.sub(r'(₹|rs\.?|inr|rupees?|years?|yrs?|old|%|percent)', '', t).replace(',', '').strip()
    m = re.fullmatch(r'([-+]?\d*\.?\d+)\s*(k|thousand|l|lakh|lakhs|lac|lacs)?', t)
    if not m:
        return None
    mult = {'k': 1e3, 'thousand': 1e3, 'l': 1e5, 'lakh': 1e5, 'lakhs': 1e5, 'lac': 1e5, 'lacs': 1e5}.get(m.group(2), 1)
    return float(m.group(1)) * mult


def validate(field, raw):
    """Return (value, error). Only the basic impossibility rules; possible-but-unusual values are accepted."""
    if field == 'make':
        text = str(raw).strip()
        return (text, None) if text else (None, 'Please tell me which vehicle you are buying.')
    if field == 'past_loans':
        t = str(raw).strip().lower()
        if t in YES or t.startswith('yes'):
            return True, None
        if t in NO or t.startswith('no'):
            return False, None
        return None, 'Please answer yes or no.'
    v = parse_number(raw)
    if v is None:
        return None, f'"{raw}" is not a number.'
    if field == 'age' and not (0 < v <= 120):
        return None, 'Age must be more than 0 and at most 120.'
    if field in ('loan_amount', 'vehicle_price') and v <= 0:
        return None, f'The {LABEL[field]} must be more than 0.'
    if field == 'ltv' and v < 0:
        return None, 'The loan-to-value ratio cannot be negative.'
    return (round(v) if field == 'age' else v), None


def rule_extract(text, asking):
    """Fallback reader when no LLM is available: the reply answers the field that was asked."""
    out = dict.fromkeys(EXTRACT_SCHEMA['properties'])
    if asking:
        out[asking] = text.strip()
    return out


# ---------------------------------------------------------------- state machine
def new_state():
    return {'values': dict.fromkeys(ORDER + ['ltv']), 'ltv_source': None, 'asking': 'make',
            'done': False, 'decision': None, 'turn': 0, 'started': time.time()}


def next_field(state):
    v = state['values']
    for f in ORDER:
        if f == 'vehicle_price' and v['ltv'] is not None and state['ltv_source'] == 'direct':
            continue
        if v[f] is None:
            return f
    return None


def fmt(field, val):
    if field in ('loan_amount', 'vehicle_price'):
        return f'₹{val:,.0f}'
    if field == 'ltv':
        return f'{val:.1f}%'
    if field == 'past_loans':
        return 'yes' if val else 'no'
    return str(val)


def process(state, user_text, llm):
    """Read one applicant message, update the state, and return a plan describing what happens next."""
    t0 = time.perf_counter()
    state['turn'] += 1
    masked = mask_pii(user_text)
    plan = {'turn': state['turn'], 'user_masked': masked, 'asked': state['asking'], 'extract_ms': 0.0}
    if state['done']:
        plan.update(action='post_decision', extractor='none')
        return plan

    ext, plan['extractor'] = None, 'rules'
    # masked placeholders make the LLM refuse to extract anything, so drop them from the text it reads
    for_llm = re.sub(r'\s*\[(?:PHONE|EMAIL|AADHAAR|PAN|NAME)\]', '', masked).strip() or masked
    if llm is not None and llm.available:
        try:
            te = time.perf_counter()
            ext = llm.json([{'role': 'system', 'content': EXTRACT_PROMPT.format(asking=LABEL.get(state['asking'], 'nothing'))},
                            {'role': 'user', 'content': for_llm}], EXTRACT_SCHEMA, name='fields')
            plan['extract_ms'] = (time.perf_counter() - te) * 1000
            plan['extractor'] = 'llm'
        except Exception as e:  # no credits / network: keep the conversation going on rules
            plan['extractor'] = f'rules (LLM unavailable: {str(e)[:60]})'
    if ext is None:
        ext = rule_extract(masked, state['asking'])
    ext = {k: (None if str(val).strip().lower() in ('', 'null', 'none') else val) for k, val in ext.items()}
    # the LLM drops unusable answers ("abc"); hand a short non-question reply to the asked field so it gets re-prompted
    if not any(ext.values()) and state['asking'] and len(masked.split()) <= 4 and not masked.strip().endswith('?'):
        ext[state['asking']] = masked.strip()
    plan['extracted'] = {k: val for k, val in ext.items() if val is not None}

    captured, errors, v = {}, [], state['values']
    for f in ['make', 'loan_amount', 'vehicle_price', 'ltv', 'age', 'past_loans']:
        raw = ext.get(f)
        if raw in (None, ''):
            continue
        val, err = validate(f, raw)
        if err:
            errors.append((f, raw, err))
            continue
        v[f], captured[f] = val, val
        if f == 'ltv':
            state['ltv_source'] = 'direct'
    if state['ltv_source'] != 'direct' and v['loan_amount'] and v['vehicle_price']:
        v['ltv'] = round(v['loan_amount'] / v['vehicle_price'] * 100, 2)
        state['ltv_source'] = 'computed'
    plan['captured'], plan['errors'] = captured, [e[2] for e in errors]

    if errors:
        f, raw, err = errors[0]
        state['asking'] = f
        plan.update(action='reask', field=f, error=err)
    elif (nf := next_field(state)):
        state['asking'] = nf
        plan.update(action='ask', field=nf, no_answer=not captured)
    else:
        _decide(state, plan, t0)
    plan['state'] = {k: val for k, val in v.items()}
    return plan


def _decide(state, plan, t0):
    v = state['values']
    row = pd.DataFrame([{'LTV': v['ltv'], 'Loan_Amount': v['loan_amount'], 'Age': v['age'],
                         'PAST_LOANS_ACTIVE': 'PAST_LOANS_ACTIVE' if v['past_loans'] else 'NO_PAST_LOANS',
                         'Make_Code': v['make']}])
    tm = time.perf_counter()
    out, X = predict.score_detailed(row)
    model_ms = (time.perf_counter() - tm) * 1000
    decision = out['Predicted_Decision'].iat[0]
    state.update(done=True, decision=decision, asking=None)
    plan.update(action='decision', decision=decision, probability=float(out['Decline_Probability'].iat[0]),
                threshold=predict.THRESHOLD, features=X.iloc[0].to_dict(), make_used=predict.make_used(X).iat[0],
                warnings=out['Data_Warnings'].iat[0], model_ms=model_ms,
                decision_ms=(time.perf_counter() - t0) * 1000,
                session_seconds=time.time() - state['started'])


# ---------------------------------------------------------------- wording the reply
def instruction(plan, state):
    a = plan['action']
    if a == 'ask':
        ins = {'task': 'acknowledge briefly, then ask the next question',
               'just_captured': {LABEL[k]: fmt(k, val) for k, val in plan['captured'].items()},
               'question': QUESTION[plan['field']]}
        if plan.get('no_answer'):
            ins['task'] = ('nothing usable was recorded from what the applicant wrote; do not claim you noted anything; '
                           'reply to what they said in one short sentence if it is about this loan application, then '
                           'ask the question again')
            ins['applicant_said'] = plan['user_masked']
        return ins
    if a == 'reask':
        return {'task': ('in one natural sentence explain why the answer cannot be used (based on "problem", do not '
                         'write the word "problem"), then ask the question again'),
                'field': LABEL[plan['field']], 'problem': plan['error'], 'question': QUESTION[plan['field']]}
    if a == 'decision':
        return {'task': 'announce the decision and the next steps; end without asking a question',
                'decision': plan['decision'],
                'next_steps': NEXT_STEPS[plan['decision']]}
    return {'task': ('answer the follow-up briefly; the decision is final for this application; to apply again the '
                     'applicant can click "New application" in the sidebar'),
            'decision': state['decision'], 'applicant_said': plan['user_masked']}


def template(plan, state):
    a = plan['action']
    if a == 'ask':
        ack = ', '.join(f'{LABEL[k]}: {fmt(k, val)}' for k, val in plan['captured'].items())
        lead = f'Thanks, noted ({ack}). ' if ack else ("Sorry, I didn't catch that. " if plan.get('no_answer') else '')
        return lead + QUESTION[plan['field']]
    if a == 'reask':
        return f"Sorry, I can't accept that: {plan['error']} {QUESTION[plan['field']]}"
    if a == 'decision':
        word = 'approved' if plan['decision'] == 'APPROVED' else 'declined'
        return f"Your loan application is {word}. {NEXT_STEPS[plan['decision']]}"
    return (f"Your application has already been {state['decision'].lower()}. To apply again, click "
            '"New application" in the sidebar.')


def reply_stream(plan, state, llm):
    """Yield the reply text: worded by the LLM when available, otherwise a fixed template.
    Whenever the vehicle is being asked for, the list of available models is appended verbatim."""
    yield from _reply_text(plan, state, llm)
    if plan.get('action') in ('ask', 'reask') and plan.get('field') == 'make':
        yield OPTIONS_MD


def _reply_text(plan, state, llm):
    if llm is not None and llm.available and not plan.get('extractor', '').startswith('rules ('):
        try:
            gen = llm.stream([{'role': 'system', 'content': COMPOSE_PROMPT},
                              {'role': 'user', 'content': 'INSTRUCTION: ' + json.dumps(instruction(plan, state), ensure_ascii=False)}])
            first = next(gen)          # fail fast before anything is shown
            yield first
            yield from gen
            return
        except Exception:
            pass
    yield template(plan, state)
