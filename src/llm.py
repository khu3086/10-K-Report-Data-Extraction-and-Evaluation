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
        _client = OpenAI(base_url=BASE_URL, api_key=api_key)
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
JSON_SPEC = (
    'Respond ONLY with a JSON object of this exact shape:\n'
    '{\n'
    '  "detected_scale": "millions" | "thousands" | "units",\n'
    '  "items": [ { "key": "<segment/region name, or \\"value\\" for a scalar>", '
    '"value": <number in ACTUAL US DOLLARS> } ]\n'
    '}'
)


def extract_field(field: Field, context: str, cycle_hint: str) -> Optional[Extraction]:
    """Call the LLM to extract `field` from `context`. Returns None on failure."""
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
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=4096,
        )
        raw = resp.choices[0].message.content or "{}"
        return Extraction.model_validate(json.loads(raw))
    except Exception as e:  # network / parse / validation errors
        print(f"  [llm] extraction failed for {field.key}: {e}")
        return None
