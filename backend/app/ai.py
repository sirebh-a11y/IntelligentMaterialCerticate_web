import json
from typing import Dict, Any, List

from openai import OpenAI

from .settings import OPENAI_MODEL


SYSTEM_PROMPT = """You extract fields from OCR tokens of a PDF certificate.
Rules:
- Do not alter token text.
- Return JSON only.
- For each field, include token_ids used (list of ints).
Fields:
- azienda
- data_certificato
- materiale
- trattamento_termico
- composizione_chimica (table with rows: {elemento, valore, token_ids})
- proprieta_meccaniche
If unknown, return empty string and empty token_ids.
"""


def _sort_tokens_for_prompt(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(tokens, key=lambda t: (t["bbox"][1], t["bbox"][0]))


def extract_fields_from_tokens(tokens: List[Dict[str, Any]], api_key: str = None) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    ordered = _sort_tokens_for_prompt(tokens)
    payload = {
        "tokens": [
            {"id": t["id"], "text": t["text"], "bbox": t["bbox"]}
            for t in ordered
        ]
    }

    input_text = (
        SYSTEM_PROMPT
        + "\nTOKENS_JSON:\n"
        + json.dumps(payload, ensure_ascii=False)
    )

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=input_text,
    )

    # Best-effort parse JSON from the model response
    text = resp.output_text
    text = text.strip()
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        text = text[first : last + 1]
    return json.loads(text)
