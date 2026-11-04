"""
Real structure prediction for generated candidates.

Full AlphaFold2 needs a multi-gigabyte sequence-database search (MSA) plus a
GPU cluster, which isn't something this dashboard can run locally. ESMFold
is the practical, widely-used stand-in: a single-sequence structure
predictor (no MSA step) from the same line of research. This module gets a
*real* predicted 3-D structure for a candidate sequence, with a per-residue
confidence (pLDDT) exactly like AlphaFold's own output uses.

Why there is more than one provider
-----------------------------------
This used to call one endpoint, ESM Atlas's free public API, and nothing
else. When that service stopped answering, every fold failed after ~100
seconds of retries with a message advising a shorter candidate -- advice
that could not help, because the service was failing at *every* length.
Measured directly against the live API: a 26 aa peptide, a 60 aa fragment
and a 117 aa antibody chain all returned `504 Endpoint request timed out`
at 29.3s, the gateway's own limit, while `esmatlas.com` itself served
normally. That is a backend outage, not a sequence problem, and no
client-side retry can fix it.

So folding now goes through an ordered cascade of providers, each tried in
turn until one returns a structure:

1. **ESM Atlas** (`esmatlas`) -- keyless, no setup, the historical default.
2. **NVIDIA NIM** (`nvidia`) -- the same ESMFold model hosted by NVIDIA.
   Needs a free API key in `NVIDIA_API_KEY`; without one it is skipped.
3. **Local ESMFold** (`local`) -- runs on this machine through
   `transformers`. Skipped unless the package is installed and
   `IA_LOCAL_ESMFOLD=1` is set, since it downloads ~2.6 GB of weights and
   wants more VRAM than this project's 50% GPU budget usually allows.

A provider that fails is remembered as unavailable for `OUTAGE_TTL_SECONDS`,
so a second click fails in milliseconds instead of waiting through the same
dead endpoint again.

When every provider is unavailable the dashboard falls back to the schematic
Chou-Fasman cartoon in `mock_results.schematic_backbone` (clearly labeled as
illustrative, not a real prediction).
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import requests
import streamlit as st

ESMFOLD_API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
NVIDIA_ESMFOLD_URL = "https://health.api.nvidia.com/v1/biology/nvidia/esmfold"

# ESM Atlas's API Gateway enforces a hard ~29s integration timeout and
# returns 504 ("Endpoint request timed out") past it, independent of
# whatever client-side `timeout` is passed. A 504/502/503 is worth one
# retry (it can be transient congestion), but not many: when the backend is
# down, every attempt costs the full 29s before failing.
RETRYABLE_STATUS_CODES = {502, 503, 504}

# How long a failed provider is treated as unavailable. Long enough that
# clicking again doesn't re-pay the timeout, short enough that a service
# coming back is picked up without restarting the app.
OUTAGE_TTL_SECONDS = 300.0

# Per-provider sequence limits live on each `Provider` below; ESM Atlas caps
# at 400 aa while NVIDIA NIM accepts up to 1024.

# Official AlphaFold/ESMFold per-residue confidence (pLDDT) color bins.
PLDDT_BANDS = [
    (90, 100, "#0053D6", "Very high"),
    (70, 90, "#65CBF3", "Confident"),
    (50, 70, "#FFDB13", "Low"),
    (0, 50, "#FF7D45", "Very low"),
]


class FoldingError(RuntimeError):
    """Raised when no provider could return a structure."""


@dataclass
class FoldResult:
    sequence: str
    pdb_text: str
    per_residue_plddt: List[float]  # always normalized to 0-100 here
    raw_bfactor_is_0_to_1: bool  # whether pdb_text's own B-factor column needs *100 to match
    provider: str = "esmatlas"  # which provider actually produced this

    @property
    def mean_plddt(self) -> float:
        return sum(self.per_residue_plddt) / len(self.per_residue_plddt) if self.per_residue_plddt else 0.0


# --------------------------------------------------------------------------
# Outage memo
# --------------------------------------------------------------------------
# Module-level rather than `st.session_state`, so one user's discovery that a
# provider is down also spares every other session the same wait.
_recent_failures: Dict[str, Tuple[float, str, int]] = {}


def _note_failure(provider: str, reason: str, length: int = 0) -> None:
    """Remember a failure, together with the sequence length it happened at."""
    _recent_failures[provider] = (time.time(), reason, length)


def _recent_failure(provider: str, length: int = 0) -> Optional[str]:
    """
    A remembered failure that should stop us trying `provider` again, if any.

    Length matters. The public ESMFold API fails in two different ways: it
    goes down for everything, and -- because its gateway cuts requests off at
    ~29s -- it also times out on sequences that simply take too long to fold.
    Measured: a 55 aa peptide came back in 0.4s while a 215 aa candidate
    504'd twice. So a failure at 215 aa says nothing about a 60 aa candidate,
    and skipping the short one because the long one failed would disable a
    working path. Only a failure at or below the requested length counts.
    """
    entry = _recent_failures.get(provider)
    if entry is None:
        return None
    when, reason, failed_at = entry
    if time.time() - when > OUTAGE_TTL_SECONDS:
        del _recent_failures[provider]
        return None
    if failed_at and length and length < failed_at:
        return None
    return reason


def reset_outage_memo() -> None:
    """Forget remembered failures, so the next fold retries every provider."""
    _recent_failures.clear()


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------
def _fold_via_esmatlas(sequence: str, timeout: float, max_attempts: int, backoff: float) -> str:
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(ESMFOLD_API_URL, data=sequence, timeout=timeout)
            response.raise_for_status()
            return response.text
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status not in RETRYABLE_STATUS_CODES:
                raise FoldingError(f"HTTP {status} from ESM Atlas ({exc}).") from exc
            last_exc = exc
        except requests.RequestException as exc:
            # Timeouts and connection-level failures (DNS, reset, ...).
            last_exc = exc
        if attempt < max_attempts:
            time.sleep(backoff * attempt)
    raise FoldingError(f"unreachable after {max_attempts} attempt(s): {last_exc}")


def _looks_like_pdb(text: object) -> bool:
    return isinstance(text, str) and ("ATOM  " in text or text.lstrip().startswith(("HEADER", "MODEL")))


def _extract_pdb_text(payload: object) -> Optional[str]:
    """
    Find the PDB text in a NIM response, whatever key it arrived under.

    NVIDIA's own pages describe this response inconsistently -- `pdbs` as a
    list, `structures` as a list, and bare `pdb`/`structure`/`data` strings
    all appear in the documentation and in third-party wrappers. Rather than
    bet on one spelling and fail with "no structure in response" against a
    perfectly good reply, search the plausible keys and then fall back to any
    string in the payload that actually looks like a PDB file.
    """
    if _looks_like_pdb(payload):
        return payload  # type: ignore[return-value]
    if isinstance(payload, list):
        for item in payload:
            found = _extract_pdb_text(item)
            if found:
                return found
        return None
    if isinstance(payload, dict):
        for key in ("pdb", "pdbs", "structure", "structures", "data", "output", "result"):
            if key in payload:
                found = _extract_pdb_text(payload[key])
                if found:
                    return found
        # Last resort: any nested value that parses as a PDB.
        for value in payload.values():
            found = _extract_pdb_text(value)
            if found:
                return found
    return None


def _nvidia_api_key() -> Optional[str]:
    return os.environ.get("NVIDIA_API_KEY") or os.environ.get("IA_NVIDIA_API_KEY")


def _fold_via_nvidia(sequence: str, timeout: float, *_unused) -> str:
    api_key = _nvidia_api_key()
    if not api_key:
        raise FoldingError("no NVIDIA_API_KEY set.")
    try:
        response = requests.post(
            NVIDIA_ESMFOLD_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json={"sequence": sequence},
            # This service has no 29s gateway cap, so give a long chain the
            # time it needs rather than the ESM Atlas-shaped timeout.
            timeout=max(timeout, 120.0),
        )
    except requests.RequestException as exc:
        raise FoldingError(f"request failed: {exc}") from exc

    if response.status_code == 401:
        raise FoldingError("NVIDIA_API_KEY was rejected (HTTP 401) -- check or regenerate it.")
    if response.status_code == 429:
        raise FoldingError("rate limited by NVIDIA (HTTP 429) -- free-tier credits may be spent.")
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise FoldingError(f"HTTP {response.status_code}: {response.text[:200]}") from exc

    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError):
        # Some deployments return the PDB file directly, not wrapped in JSON.
        payload = response.text

    pdb_text = _extract_pdb_text(payload)
    if not pdb_text:
        shape = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        raise FoldingError(f"no structure found in the response (shape: {shape}).")
    return pdb_text


def _fold_locally(sequence: str, *_unused) -> str:
    if os.environ.get("IA_LOCAL_ESMFOLD", "").lower() not in {"1", "true", "yes"}:
        raise FoldingError("disabled (set IA_LOCAL_ESMFOLD=1 to enable).")
    try:
        import torch
        from transformers import AutoTokenizer, EsmForProteinFolding
    except ImportError as exc:
        raise FoldingError(
            "`transformers` is not installed (uv add transformers accelerate)."
        ) from exc

    model, tokenizer = _load_local_esmfold()
    inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return model.output_to_pdb(outputs)[0]


@st.cache_resource(show_spinner="Loading local ESMFold weights (~2.6 GB, first run only)...")
def _load_local_esmfold():
    """Load ESMFold once per process; the weights are large and slow to read."""
    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding

    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        # Chunked attention keeps peak VRAM inside this project's 50% GPU
        # budget; without it ESMFold wants well over the 4 GB available.
        model.trunk.set_chunk_size(64)
    return model, tokenizer


@dataclass
class Provider:
    name: str
    label: str
    fold: Callable[..., str]
    is_configured: Callable[[], bool]
    max_length: int
    needs: str = ""  # what the user must do to enable it, if anything


# Order matters: the first provider that is configured AND can take the
# sequence wins. NVIDIA NIM goes first because it is only ever configured
# deliberately (it needs a key) and, unlike ESM Atlas, has no ~29s gateway
# cap -- which is the whole reason a full-length antibody chain fails there.
# ESM Atlas stays as the keyless default for anyone who sets nothing up.
PROVIDERS: List[Provider] = [
    Provider(
        "nvidia", "NVIDIA NIM ESMFold", _fold_via_nvidia,
        is_configured=lambda: bool(_nvidia_api_key()),
        max_length=1024,
        needs="set NVIDIA_API_KEY (free key from build.nvidia.com/nvidia/esmfold)",
    ),
    Provider(
        "esmatlas", "ESM Atlas (free public API)", _fold_via_esmatlas,
        is_configured=lambda: True,
        max_length=400,
    ),
    Provider(
        "local", "Local ESMFold (this machine)", _fold_locally,
        is_configured=lambda: os.environ.get("IA_LOCAL_ESMFOLD", "").lower() in {"1", "true", "yes"},
        max_length=1024,
        needs="install transformers and set IA_LOCAL_ESMFOLD=1",
    ),
]


def max_foldable_length() -> int:
    """Longest sequence any configured provider will accept."""
    lengths = [p.max_length for p in PROVIDERS if p.is_configured()]
    return max(lengths) if lengths else 0


def provider_status() -> List[Dict[str, str]]:
    """
    One row per provider: whether it is configured, and any recent failure.

    The dashboard shows this so an outage is visible *before* clicking, and
    so it is obvious that the two fallbacks exist but are not set up.
    """
    rows = []
    for provider in PROVIDERS:
        rows.append({
            "name": provider.name,
            "label": provider.label,
            "configured": "yes" if provider.is_configured() else "no",
            "needs": provider.needs,
            "max_length": str(provider.max_length),
            "recent_failure": _recent_failure(provider.name) or "",
        })
    return rows


# --------------------------------------------------------------------------
# PDB parsing
# --------------------------------------------------------------------------
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
    sequence: str, timeout: float = 35.0, max_attempts: int = 2, retry_backoff: float = 2.0
) -> FoldResult:
    """
    Predict a 3-D structure for `sequence`, trying each provider in turn.

    Cached by sequence (an hour at a time) so revisiting the same candidate
    doesn't re-hit the network. Raises `FoldingError` listing what each
    provider said -- callers should catch it and fall back to the schematic
    cartoon.

    Providers that failed within the last `OUTAGE_TTL_SECONDS` are skipped
    rather than retried, which is what keeps a dead endpoint from costing
    ~30s per attempt on every click.
    """
    clean_sequence = re.sub(r"[^A-Za-z]", "", sequence).upper()
    if not clean_sequence:
        raise FoldingError("Empty sequence.")
    longest = max_foldable_length()
    if longest and len(clean_sequence) > longest:
        raise FoldingError(
            f"Sequence is {len(clean_sequence)} aa; the longest any configured provider "
            f"accepts is {longest} aa. Try a shorter candidate, or configure a provider "
            "with a higher limit (NVIDIA NIM takes up to 1024 aa)."
        )

    attempted: List[str] = []
    length = len(clean_sequence)
    for provider in PROVIDERS:
        if not provider.is_configured():
            attempted.append(f"{provider.label}: not set up ({provider.needs})")
            continue
        if length > provider.max_length:
            attempted.append(
                f"{provider.label}: declines {length} aa (its limit is {provider.max_length} aa)"
            )
            continue
        remembered = _recent_failure(provider.name, length)
        if remembered is not None:
            attempted.append(
                f"{provider.label}: skipped, failed at this length within the last "
                f"{int(OUTAGE_TTL_SECONDS / 60)} min ({remembered})"
            )
            continue
        try:
            pdb_text = provider.fold(clean_sequence, timeout, max_attempts, retry_backoff)
        except FoldingError as exc:
            _note_failure(provider.name, str(exc), length)
            attempted.append(f"{provider.label}: {exc}")
            continue
        except Exception as exc:  # a provider must never take the app down
            _note_failure(provider.name, str(exc), length)
            attempted.append(f"{provider.label}: {type(exc).__name__}: {exc}")
            continue

        plddt_raw = _parse_ca_bfactors(pdb_text)
        if not plddt_raw:
            _note_failure(provider.name, "returned no parseable structure", length)
            attempted.append(f"{provider.label}: returned no parseable structure")
            continue

        # Some providers report pLDDT on a 0-1 scale rather than AlphaFold's
        # usual 0-100; detect and normalize so downstream color bins/metrics
        # are always on the standard 0-100 scale. `pdb_text` itself is left
        # as-is -- `raw_bfactor_is_0_to_1` tells a renderer reading straight
        # from that text how to interpret it.
        raw_is_0_to_1 = max(plddt_raw) <= 1.5
        plddt = [v * 100 for v in plddt_raw] if raw_is_0_to_1 else plddt_raw
        # It works again: forget any remembered failure so a longer candidate
        # is not skipped on the strength of stale bad news.
        _recent_failures.pop(provider.name, None)
        return FoldResult(
            sequence=clean_sequence, pdb_text=pdb_text, per_residue_plddt=plddt,
            raw_bfactor_is_0_to_1=raw_is_0_to_1, provider=provider.name,
        )

    # Deliberately factual about what was tried. The previous single-provider
    # message blamed the sequence ("try a shorter candidate") for what is
    # usually a service outage affecting every length equally.
    detail = "; ".join(attempted)
    raise FoldingError(
        f"No folding provider could return a structure for this {len(clean_sequence)} aa "
        f"sequence. Tried -- {detail}. "
        "The free ESM Atlas API is unreliable in two ways at once: it goes down for "
        "everything from time to time, and its gateway cuts every request off at ~29s, "
        "which a full-length antibody chain often cannot beat (measured: 55 aa folded in "
        "0.4s, 215 aa timed out twice). Retrying may work; a shorter candidate may work. "
        "For a path that does not depend on that service, set up one of the fallback "
        "providers -- a free NVIDIA_API_KEY, or a local ESMFold install. The schematic "
        "cartoon above is still a valid illustration in the meantime."
    )


def plddt_band_color(plddt: float) -> str:
    for lo, hi, color, _label in PLDDT_BANDS:
        if lo <= plddt <= hi:
            return color
    return PLDDT_BANDS[-1][2]
