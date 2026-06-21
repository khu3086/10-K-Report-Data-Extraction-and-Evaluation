"""LLM client wrapper for structured field extraction (OpenRouter).

OpenRouter exposes an OpenAI-compatible API, so we use the OpenAI SDK pointed at
the OpenRouter base URL. The model returns a JSON object matching the Extraction
schema, which we validate with Pydantic.

Config (env):
  OPENROUTER_API_KEY   API key (sk-or-...)   [required]
  EXTRACT_MODEL        model id              [default: openai/gpt-4o-mini]
  OPENAI_BASE_URL      override base url     [default: https://openrouter.ai/api/v1]
"""

import json
import os
from typing import List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field as PydField

from fields import Field

MODEL = os.environ.get("EXTRACT_MODEL", "openai/gpt-4o-mini")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENROUTER_API_KEY (or OPENAI_API_KEY).")
        # Bounded timeout + a couple retries so a single slow/rate-limited free
        # endpoint can't stall the whole batch (the call site treats None as skip).
        _client = OpenAI(base_url=BASE_URL, api_key=api_key,
                         timeout=float(os.environ.get("LLM_TIMEOUT", "90")),
                         max_retries=int(os.environ.get("LLM_MAX_RETRIES", "2")))
    return _client


# --- Structured output schema --------------------------------------------------
# Uniform schema for both scalar and keyed fields. Scalar fields come back as a
# single item with key "value"; keyed fields as one item per segment/region.
# `value` is always in ACTUAL DOLLARS.

class Item(BaseModel):
    key: str = PydField(description="Segment/region name, or 'value' for scalar fields")
    # Optional so a single null from the model doesn't invalidate the whole
    # extraction; null items are filtered out downstream.
    value: Optional[float] = PydField(default=None, description="Amount in actual US dollars")


class Extraction(BaseModel):
    detected_scale: str = PydField(
        description="The scale stated in the source table: 'millions', "
        "'thousands', or 'units' if none."
    )
    items: List[Item] = PydField(description="One entry per extracted value")


SYSTEM = (
    "You are a precise financial-data extraction system. You read excerpts of "
    "SEC 10-K filings and return numeric values exactly as instructed. You never "
    "fabricate figures: if a value is not present in the excerpt, you omit it. "
    "You always respond with a single JSON object and nothing else."
)

# The exact JSON shape we expect back (kept in the prompt so any OpenRouter
# model — not just those supporting strict structured outputs — can comply).
# Whether `value` is "as printed" or "actual dollars" is governed by the
# cycle hint passed in (see fields.CYCLE_HINTS); detected_scale is always the
# scale stated in the source table.
JSON_SPEC = (
    'Respond ONLY with a JSON object of this exact shape:\n'
    '{\n'
    '  "detected_scale": "millions" | "thousands" | "units",\n'
    '  "items": [ { "key": "<segment/region name, or \\"value\\" for a scalar>", '
    '"value": <number> } ]\n'
    '}'
)


def _clean_json(raw: str) -> str:
    """Strip markdown fences / prose so json.loads sees a bare object.

    Some models wrap JSON in ```json fences or add a sentence; we keep the
    substring from the first '{' to the last '}'.
    """
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.lstrip().lower().startswith("json"):
            s = s.lstrip()[4:]
    i, j = s.find("{"), s.rfind("}")
    return s[i:j + 1] if (i != -1 and j != -1 and j > i) else s


def extract_field(field: Field, context: str, cycle_hint: str,
                  model: Optional[str] = None) -> Optional[Extraction]:
    """Call the LLM to extract `field` from `context`. Returns None on failure.

    `model` overrides the default model (used by the ground-truth builder to run
    a stronger, independent model than the system under test).
    """
    shape = ("one value per segment/region" if field.shape == "keyed"
             else "a single scalar value (key = 'value')")
    user = (
        f"{cycle_hint}\n\n"
        f"FIELD: {field.label}\n"
        f"WHAT TO EXTRACT: {field.description}\n"
        f"SHAPE: {shape}\n\n"
        f"{JSON_SPEC}\n\n"
        f"10-K EXCERPT:\n{context}"
    )
    try:
        resp = _get_client().chat.completions.create(
            model=model or MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=2000,
        )
        raw = resp.choices[0].message.content or "{}"
        return Extraction.model_validate(json.loads(_clean_json(raw)))
    except Exception as e:  # network / parse / validation errors
        print(f"  [llm] extraction failed for {field.key}: {e}")
        return None
