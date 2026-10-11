"""
Real structure prediction for generated candidates.

Full AlphaFold2 needs a multi-gigabyte sequence-database search (MSA) plus a
GPU cluster, which isn't something this dashboard can run locally. ESMFold
is the practical, widely-used stand-in: a single-sequence structure
predictor (no MSA step) from the same line of research, offered for free as
a public API by the ESM Atlas project. This module calls that API to get a
*real* predicted 3-D structure for a candidate sequence, with a per-residue
confidence (pLDDT) exactly like AlphaFold's own output uses.

This is used on-demand (a button click), not automatically for every
candidate, both because it depends on an internet connection and to avoid
hammering a free public API with a full batch of candidates at once. When
it's unavailable, the dashboard falls back to the schematic Chou-Fasman
cartoon in `mock_results.schematic_backbone` (clearly labeled as illustrative,
not a real prediction).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import List, Optional

import requests
import streamlit as st

ESMFOLD_API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"

# ESM Atlas's own API Gateway in front of this endpoint enforces a hard
# ~29s integration timeout and returns 504 ("Endpoint request timed out")
# past it -- independent of whatever client-side `timeout` is passed below.
# Measured directly against the live API: a 90+ aa sequence 504'd three
# times in a row under load, then succeeded in ~1s moments later once that
# load passed. So a 504/502/503 here is normally transient server-side
# congestion, not a sequence the API can never fold -- worth retrying.
RETRYABLE_STATUS_CODES = {502, 503, 504}

# Official AlphaFold/ESMFold per-residue confidence (pLDDT) color bins.
PLDDT_BANDS = [
    (90, 100, "#0053D6", "Very high"),
    (70, 90, "#65CBF3", "Confident"),
    (50, 70, "#FFDB13", "Low"),
    (0, 50, "#FF7D45", "Very low"),
]


class FoldingError(RuntimeError):
    """Raised when the ESMFold API can't be reached or returns no structure."""


@dataclass
class FoldResult:
    sequence: str
    pdb_text: str
    per_residue_plddt: List[float]  # always normalized to 0-100 here
    raw_bfactor_is_0_to_1: bool  # whether pdb_text's own B-factor column needs *100 to match

    @property
    def mean_plddt(self) -> float:
        return sum(self.per_residue_plddt) / len(self.per_residue_plddt) if self.per_residue_plddt else 0.0


def _parse_ca_bfactors(pdb_text: str) -> List[float]:
    """Pull the B-factor (pLDDT) column for each residue's CA atom, in order."""
    values = []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                values.append(float(line[60:66]))
            except ValueError:
                continue
    return values


@st.cache_data(show_spinner=False, ttl=3600)
def fold_sequence(
    sequence: str, timeout: float = 40.0, max_attempts: int = 3, retry_backoff: float = 3.0
) -> FoldResult:
    """
    Predict a 3-D structure for `sequence` via the public ESMFold API.

    Cached by sequence (an hour at a time) so revisiting the same candidate
    doesn't re-hit the network. Raises `FoldingError` on any network issue,
    timeout, or empty/invalid response -- callers should catch this and fall
    back to the schematic cartoon.

    A 502/503/504 or a client-side timeout is retried up to `max_attempts`
    times with a linear backoff (`retry_backoff * attempt` seconds between
    tries), since the API gateway's own ~29s timeout means these are
    typically transient server load, not a sequence it can never fold (see
    `RETRYABLE_STATUS_CODES`). Any other HTTP error (e.g. a malformed
    request) fails immediately instead of burning retries on it.
    """
    clean_sequence = re.sub(r"[^A-Za-z]", "", sequence).upper()
    if not clean_sequence:
        raise FoldingError("Empty sequence.")
    if len(clean_sequence) > 400:
        raise FoldingError(
            f"Sequence is {len(clean_sequence)} aa; the public ESMFold API caps out well before "
            "that length. Try a shorter candidate."
        )

    pdb_text: Optional[str] = None
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(ESMFOLD_API_URL, data=clean_sequence, timeout=timeout)
            response.raise_for_status()
            pdb_text = response.text
            break
        except requests.Timeout as exc:
            last_exc = exc
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status not in RETRYABLE_STATUS_CODES:
                raise FoldingError(f"ESMFold API returned HTTP {status} ({exc}).") from exc
            last_exc = exc
        except requests.RequestException as exc:
            # Connection-level failures (DNS, reset, etc.) -- also worth a retry.
            last_exc = exc

        if attempt < max_attempts:
            time.sleep(retry_backoff * attempt)

    if pdb_text is None:
        raise FoldingError(
            f"Could not reach the ESMFold API after {max_attempts} attempt(s) "
            f"(last error: {last_exc}). This is ESM Atlas's free public API -- its gateway "
            "times out requests at ~30s and can be overloaded. It's usually transient: wait a "
            "moment and click 'Fold with ESMFold' again, or try a shorter candidate."
        ) from last_exc

    plddt_raw = _parse_ca_bfactors(pdb_text)
    if not plddt_raw:
        raise FoldingError("ESMFold API returned no parseable structure for this sequence.")

    # This API happens to report pLDDT on a 0-1 scale rather than AlphaFold's
    # usual 0-100; detect and normalize so downstream color bins/metrics are
    # always on the standard 0-100 scale. `pdb_text` itself is left as-is
    # (still whatever scale the API returned) -- `raw_bfactor_is_0_to_1`
    # tells a renderer reading straight from that text how to interpret it.
    raw_is_0_to_1 = max(plddt_raw) <= 1.5
    plddt = [v * 100 for v in plddt_raw] if raw_is_0_to_1 else plddt_raw

    return FoldResult(
        sequence=clean_sequence, pdb_text=pdb_text, per_residue_plddt=plddt,
        raw_bfactor_is_0_to_1=raw_is_0_to_1,
    )


def plddt_band_color(plddt: float) -> str:
    for lo, hi, color, _label in PLDDT_BANDS:
        if lo <= plddt <= hi:
            return color
    return PLDDT_BANDS[-1][2]
