"""Send log rows to a Google Sheet through the Apps Script web app (apps_script/Code.gs).

Posting happens on a background thread, so logging never slows the chat. Without a configured URL the logger is a
no-op (the app shows "logging off" in the sidebar).
"""
import json
import math
import threading
import time
import requests

CHUNK = 2000


def _clean(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return ''
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, ensure_ascii=False, default=str)
    if hasattr(v, 'item'):          # numpy scalars
        v = v.item()
    return v if isinstance(v, (int, float, str, bool)) else str(v)


class SheetsLogger:
    def __init__(self, url: str | None, token: str | None):
        self.url, self.token = (url or '').strip(), (token or '').strip()
        self.last_status, self.sent, self.failed = 'not used yet', 0, 0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.url.startswith('https://')

    def log(self, sheet: str, rows: list[dict], wait: bool = False):
        if not self.enabled or not rows:
            return
        rows = [{k: _clean(v) for k, v in r.items()} for r in rows]
        t = threading.Thread(target=self._post, args=(sheet, rows), daemon=True)
        t.start()
        if wait:
            t.join(timeout=120)

    def _post(self, sheet, rows):
        headers = list(dict.fromkeys(k for r in rows for k in r))
        for i in range(0, len(rows), CHUNK):
            chunk = rows[i:i + CHUNK]
            for attempt in range(3):
                try:
                    r = requests.post(self.url, timeout=60, data=json.dumps(
                        {'token': self.token, 'sheet': sheet, 'headers': headers, 'rows': chunk}, default=str),
                        headers={'Content-Type': 'text/plain;charset=utf-8'})
                    ok = r.ok and r.json().get('ok')
                    with self._lock:
                        if ok:
                            self.sent += len(chunk)
                            self.last_status = f'ok ({time.strftime("%H:%M:%S")})'
                        else:
                            self.last_status = f'error: {r.text[:120]}'
                    if ok:
                        break
                except Exception as e:
                    with self._lock:
                        self.last_status = f'error: {str(e)[:120]}'
                time.sleep(1.5 * (attempt + 1))
            else:
                with self._lock:
                    self.failed += len(chunk)
