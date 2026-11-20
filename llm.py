"""Minimal client for OpenAI-compatible Chat Completions APIs (Groq or OpenAI), via plain HTTPS.

Used only in the interaction layer: reading the applicant's answers and wording replies.
It never makes or changes the credit decision.
"""
import json
import requests

PROVIDERS = {
    'groq': {'url': 'https://api.groq.com/openai/v1/chat/completions', 'model': 'openai/gpt-oss-120b'},
    'openai': {'url': 'https://api.openai.com/v1/chat/completions', 'model': 'gpt-4.1-mini'},
}


class LLM:
    def __init__(self, api_key, provider='groq', model=None, timeout=20):
        cfg = PROVIDERS[provider]
        self.provider, self.url = provider, cfg['url']
        self.model = model or cfg['model']
        self.api_key, self.timeout = (api_key or '').strip(), timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    @property
    def name(self) -> str:
        return f'{self.provider}:{self.model}'

    def _body(self, messages, max_tokens, temperature):
        body = {'model': self.model, 'messages': messages, 'max_completion_tokens': max_tokens}
        if 'gpt-oss' in self.model:                       # reasoning model: keep thinking short and hidden
            body.update(reasoning_effort='low', include_reasoning=False, temperature=temperature)
        elif not self.model.startswith(('gpt-5', 'gpt-6', 'o1', 'o3', 'o4')):  # these reject temperature
            body['temperature'] = temperature
        return body

    def _headers(self):
        return {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}

    def _post(self, body, stream=False):
        r = requests.post(self.url, headers=self._headers(), json=body, timeout=self.timeout, stream=stream)
        if not r.ok:
            msg = r.text[:200]
            r.close()
            raise RuntimeError(f'{self.provider} error {r.status_code}: {msg}')
        return r

    def json(self, messages, schema: dict, name: str = 'result', max_tokens: int = 700) -> dict:
        body = self._body(messages, max_tokens, 0)
        body['response_format'] = {'type': 'json_schema', 'json_schema': {'name': name, 'strict': True, 'schema': schema}}
        r = self._post(body)
        return json.loads(r.json()['choices'][0]['message']['content'])

    def stream(self, messages, max_tokens: int = 700, temperature: float = 0.3):
        body = self._body(messages, max_tokens, temperature)
        body['stream'] = True
        with self._post(body, stream=True) as r:
            r.encoding = 'utf-8'
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith('data: '):
                    continue
                data = line[6:]
                if data.strip() == '[DONE]':
                    break
                choices = json.loads(data).get('choices') or [{}]
                piece = choices[0].get('delta', {}).get('content')
                if piece:
                    yield piece
