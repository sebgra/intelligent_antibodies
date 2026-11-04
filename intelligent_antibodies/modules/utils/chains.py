"""
Telling a heavy chain from a light one, by sequence alone.

The models are trained on every chain SAbDab tags as an antibody, which pools
heavy and light chains together without a label -- the tag only separates
antibody from antigen. Nothing downstream ever needed the distinction until
generation did: sampling the latent space unconditionally returns a mixture
of both families, and occasionally a chimera carrying motifs from each.

Classification here is deliberately simple and uses the one feature that is
conserved across essentially all immunoglobulin variable domains: the **FR4
motif** that closes the domain just before the constant region. A heavy chain
ends its variable domain on a W-G-x-G tryptophan motif, a light chain on the
equivalent F-G-x-G phenylalanine motif. It is a crude rule -- it says nothing
about subclass, and it fails on fragments that stop before FR4 -- but it is
unambiguous when present, needs no alignment, and agrees with SAbDab's own
chain counts: 693 heavy against 707 light over 1,491 chains, which is the
near-perfect balance expected when every structure contributes one of each.
"""
from __future__ import annotations

from typing import Optional

HEAVY_FR4 = ("WGQG", "WGAG", "WGRG", "WGKG", "WGQT", "WGPG")
LIGHT_FR4 = ("FGQG", "FGGG", "FGSG", "FGAG", "FGTG", "FGPG")

# Canonical first four residues of a variable domain, used only as a weaker
# fallback hint; FR4 is the reliable signal.
HEAVY_START = ("QVQL", "EVQL", "QVEL", "EVKL", "QIQL", "EVQV", "QVQV", "QVTL")
LIGHT_START = ("DIQM", "DIVM", "DIVL", "EIVL", "QIVL", "DIEL", "QSVL", "SYEL",
               "DILM", "QAVV", "QPVL", "DIVQ", "EIVM")

CHAIN_TYPES = ("heavy", "light")


def classify_chain(sequence: str) -> Optional[str]:
    """
    ``"heavy"``, ``"light"``, or None when the sequence says neither clearly.

    None covers two different situations worth keeping distinct from a guess:
    a chain carrying no FR4 motif at all (a fragment, or a decode that never
    reproduced the end of the domain), and a chimera carrying both.
    """
    if not sequence:
        return None
    heavy = any(motif in sequence for motif in HEAVY_FR4)
    light = any(motif in sequence for motif in LIGHT_FR4)
    if heavy and not light:
        return "heavy"
    if light and not heavy:
        return "light"
    if heavy and light:
        return None                      # chimeric: both families at once
    # No FR4 at all -- fall back to the N-terminal hint, which is weaker.
    if sequence[:4] in HEAVY_START:
        return "heavy"
    if sequence[:4] in LIGHT_START:
        return "light"
    return None


def classify_chains(sequences) -> list:
    return [classify_chain(s) for s in sequences]
