# ABC Credit: two-wheeler loan approval chatbot

A Streamlit app that gives an **Approve / Decline** decision on a two-wheeler loan.

- **Chat application**: a short conversation that asks only the 5 inputs the model needs.
- **Batch scoring**: upload an Excel or CSV test file. The app matches its columns to the model inputs, scores every
  row and returns the file with the decision appended as the **last column** (`Predicted_Decision`).

The decision comes **only from the trained XGBoost model** (a decline probability compared against a threshold of 0.53).
The language model (Groq `openai/gpt-oss-120b`, or OpenAI) works only in the interaction layer: it reads answers into
fields, words the replies, and suggests column matches for uploaded files. It never makes or changes a decision.

## How it works

| Step | Component |
|---|---|
| Read the applicant's answer into fields | LLM with strict JSON output (`chat_engine.py`); simple rules if no LLM is available |
| Validate inputs | Python. Only basic rules: age must be above 0 and at most 120, loan amount and price must be above 0, LTV cannot be negative, numbers must be numbers. Invalid answers are asked again |
| Encode, clip to training ranges, map unseen makes to `OTHER` | `pipeline.py` |
| Approve / decline | `predict.py`: XGBoost probability ≥ 0.53 → DECLINED |
| Word the reply and next steps | LLM, constrained to the decision it is given; fixed wording if no LLM is available |
| Mask personal data | `pii.py` masks phone, email, Aadhaar, PAN and names before anything is logged or sent to the LLM |
| Log every turn and batch row | `sheets_logger.py` → Google Apps Script (`apps_script/Code.gs`) → Google Sheet |

**Questions asked (5):** vehicle make/model, loan amount, on-road price (used to compute LTV; skipped if the applicant
states the LTV), age, and active past loans. These are the model's features: LTV, Loan_Amount, Age, PAST_LOANS_ACTIVE
and Make_Code. Several answers can be given in one message.

**Measured speed (local run, Groq `gpt-oss-120b`):**
- About 1.5 to 2.5 s per chat turn, including reading the answer and streaming the reply.
- Model scoring takes about 15 ms per application.
- Decision latency, from the final answer to a decision, is about 0.8 to 1.4 s.
- Batch scoring takes about 35 ms for 3,000 rows.

That supports targets of **X ≤ 2 minutes** to finish an application (5 short answers) and **Y ≤ 3 seconds** for the
decision.

## Run locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then fill in your keys
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to GitHub. `.streamlit/secrets.toml` is git-ignored, so **no API key is ever committed**.
2. On share.streamlit.io choose *New app*, pick the repo, and set the main file to `app.py`.
3. In *App settings → Secrets*, paste the contents of `.streamlit/secrets.toml.example` with your real values.

## Google Sheets logging

1. Create a Google Sheet, open *Extensions → Apps Script*, and paste in `apps_script/Code.gs`.
2. Set `TOKEN` in the script to a long random string.
3. Choose *Deploy → New deployment → Web app*, with *Execute as: Me* and *Who has access: Anyone*, and copy the `/exec` URL.
4. Add `GSHEET_WEBHOOK_URL` (the URL) and `GSHEET_TOKEN` (the same token) to the Streamlit secrets.

The script creates three tabs on first use:
- `chat_log`: one row per chat turn, with the masked message, the reply, extracted values, validation errors and timings. The final turn also holds the model inputs, decline probability, threshold, decision and latencies.
- `batch_runs`: one row per uploaded file, with the column mapping, counts and any metrics.
- `batch_predictions`: one row per scored record, with the raw inputs, the model inputs, probability, decision and warnings.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit user interface (chat and batch) |
| `chat_engine.py` | Question flow, validation, and the call to the model |
| `column_mapper.py` | Matches uploaded columns to model inputs (by name, then LLM) |
| `llm.py` | Minimal Groq/OpenAI chat client |
| `pii.py` | Personal-data masking |
| `sheets_logger.py`, `apps_script/Code.gs` | Google Sheets logging |
| `pipeline.py`, `predict.py` | Encoding, input checks and scoring (shared by training and the app) |
| `train_xgb.py` | Reproducible model training (reads `training_data.xlsx`) |
| `xgb_model.json`, `xgb_model_meta.json` | The trained model, threshold and encoding settings |
