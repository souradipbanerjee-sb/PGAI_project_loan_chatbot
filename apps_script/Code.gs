/**
 * ABC Credit loan chatbot - Google Sheets logger.
 *
 * Setup (once):
 *  1. Create a Google Sheet. Extensions -> Apps Script. Replace the editor contents with this file.
 *  2. Change TOKEN below to any long random string. Put the same string in the Streamlit secret GSHEET_TOKEN.
 *  3. Deploy -> New deployment -> type "Web app". Execute as: Me. Who has access: Anyone.
 *     Copy the Web app URL (ends in /exec) into the Streamlit secret GSHEET_WEBHOOK_URL.
 *  4. After editing this script later, use Deploy -> Manage deployments -> Edit -> Version: New version.
 *
 * The app sends {token, sheet, headers, rows}. Each target tab is created on first use with a header row;
 * new fields are added as new columns, so the log keeps working if the app starts logging extra values.
 * Tabs used by the app: chat_log, batch_runs, batch_predictions.
 */
const TOKEN = 'CHANGE_ME_TO_A_LONG_RANDOM_STRING';
const ALLOWED_SHEETS = ['chat_log', 'batch_runs', 'batch_predictions'];

function doPost(e) {
  let body;
  try {
    body = JSON.parse(e.postData.contents);
  } catch (err) {
    return reply_({ ok: false, error: 'bad json' });
  }
  if (body.token !== TOKEN) return reply_({ ok: false, error: 'unauthorized' });
  if (ALLOWED_SHEETS.indexOf(body.sheet) < 0) return reply_({ ok: false, error: 'unknown sheet' });
  const rows = body.rows || [];
  if (!rows.length) return reply_({ ok: true, appended: 0 });

  const lock = LockService.getScriptLock();
  lock.waitLock(30000);
  try {
    const sheet = getSheet_(body.sheet);
    const header = ensureHeader_(sheet, body.headers || Object.keys(rows[0]));
    const values = rows.map(function (r) {
      return header.map(function (h) { return r[h] === undefined || r[h] === null ? '' : r[h]; });
    });
    sheet.getRange(sheet.getLastRow() + 1, 1, values.length, header.length).setValues(values);
    return reply_({ ok: true, appended: values.length });
  } finally {
    lock.releaseLock();
  }
}

function doGet() {
  return reply_({ ok: true, service: 'ABC Credit loan chatbot logger' });
}

function getSheet_(name) {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  return ss.getSheetByName(name) || ss.insertSheet(name);
}

function ensureHeader_(sheet, wanted) {
  const lastCol = sheet.getLastColumn();
  let header = lastCol ? sheet.getRange(1, 1, 1, lastCol).getValues()[0].filter(String) : [];
  const extra = wanted.filter(function (h) { return header.indexOf(h) < 0; });
  if (extra.length) {
    sheet.getRange(1, header.length + 1, 1, extra.length).setValues([extra]).setFontWeight('bold');
    header = header.concat(extra);
    sheet.setFrozenRows(1);
  }
  return header;
}

function reply_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}
