"""
Indian number plate structure — validation and repair.

The OCR stage returns characters. This turns them into a plate, or rejects
them, by knowing what an Indian plate looks like:

    GJ 01 WA 9658
    ^^ ^^ ^^ ^^^^
    |  |  |  +--- 4 digits, the unique number
    |  |  +------ 1-3 letters, the series
    |  +--------- 1-2 digits, the RTO district
    +------------ 2 letters, the state code

Measured against every distinct string this service has read from live cameras,
49 of 68 break that pattern, and every one of those is a misread. Character
cleaning cannot tell them from a real plate. The pattern can, and can often say
what the plate should have been: OCR confuses characters that look alike, and
the position says which member of a look-alike pair belongs there. A 6 in the
state code is a G; a B in the number is an 8. One global confusion table can
never express that, because the direction depends on the position.

What this deliberately does NOT do
----------------------------------
Guess when the structure fits but the characters are ambiguous. GJ01HA9658 and
GJ01VA9658 are both well-formed, and only repeated reads can say which is the
real GJ01WA9658. Inventing an answer there would turn a visible misread into an
invisible one. Truncated reads (26BT1607) are rejected for the same reason —
the missing characters are not recoverable, and the vote already handles them
by clustering with the full read.
"""

import re

PLATE_RE = re.compile(r"^([A-Z]{2})([0-9]{1,2})([A-Z]{1,3})([0-9]{4})$")

# Every RTO state and union territory code in use. GU, 6J and GJZ are not on
# it, which is what catches misreads that are otherwise well formed.
STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK",
    "UP", "WB",
}

# Characters that look like each other, as observed in this service's own logs.
# Each cluster is bidirectional: any member can be read as any other. The
# position decides which member is admissible, so 0/O/D/Q/J/U in a digit slot
# resolves to 0, and in a letter slot to one of O, D, Q, J or U.
CLUSTERS = (
    "0ODQJU",   # G027DM7295 was GJ27..., GJONP4188 was GJ01...
    "1ILJT",    # HR26B11607 was HR26BT1607
    "2Z",       # GJZ7R8104 was GJ27...
    "4A",
    "5S",
    "6GC",      # 6J27R4290 was GJ27R4290
    "7T",
    "8BR",      # HR268T1607 was HR26BT1607
    "9PG",
    "MN",       # GJ27DN7295 was GJ27DM7295
    "VWYU",     # GJ01VA9658 was GJ01WA9658
    "XK",
    "EF",
    "HAM",      # AR26BT1607 was HR26BT1607 — the case that prompted this
)

_SIMILAR = {}
for _c in CLUSTERS:
    for _ch in _c:
        _SIMILAR.setdefault(_ch, set()).update(_c)


def _options(ch, want):
    """
    Characters this one could really be, in a slot that must hold `want`.

    Returns [(character, repair_cost)], the observed character first at zero
    cost so an unrepaired reading always wins a tie.
    """
    pool = _SIMILAR.get(ch, set()) | {ch}
    keep = str.isdigit if want == "digit" else str.isalpha

    out = []
    for c in sorted(pool):
        if keep(c):
            out.append((c, 0 if c == ch else 1))

    out.sort(key=lambda x: x[1])
    return out


def _coerce(chunk, want, budget):
    """
    Cheapest way to force a run of characters into letters or digits.

    Greedy per character: each slot independently takes its cheapest
    admissible option, which is correct here because the slots are unrelated.
    Returns (text, repairs) or (None, None) when some character has no
    admissible reading or the budget is blown.
    """
    out = []
    cost = 0

    for ch in chunk:
        opts = _options(ch, want)
        if not opts:
            return None, None
        c, n = opts[0]
        cost += n
        if cost > budget:
            return None, None
        out.append(c)

    return "".join(out), cost


def _state_options(pair, expected, penalty):
    """
    Real state codes this two-character reading could be.

    Returns [(code, repairs)] sorted cheapest first, preferring codes the site
    actually expects. This is what settles AR26BT1607: AR is a real code, so
    nothing about the plate's structure is wrong, but H and A look alike and
    HR is a code this site sees every day while AR is 2,000km away.
    """
    first = _options(pair[0], "alpha")
    second = _options(pair[1], "alpha")

    found = []
    for a, ca in first:
        for b, cb in second:
            code = a + b
            if code in STATE_CODES:
                cost = ca + cb
                if expected and code not in expected:
                    cost += penalty
                found.append((code, cost))

    found.sort(key=lambda x: x[1])
    return found


# Shortest real shape is LL D L DDDD. Anything shorter is a truncated read, and
# padding it out would invent a plate rather than recover one.
MIN_LEN = 9
MAX_LEN = 11


def parse(raw, states=None, max_repairs=2.5, state_penalty=1.5):
    """
    Read one OCR string as a plate.

    Returns a dict, or None when the string cannot be made into a plate within
    max_repairs:

        {"plate", "state", "district", "series", "number",
         "repairs", "dropped", "exact"}

    Every segmentation over every substring is tried and the cheapest wins —
    fewest repairs, then fewest characters discarded. Sliding over substrings
    is what removes the junk that plate bolts and borders add to the ends
    (FHR26BT1607, GJ278104H).

    states is the set of state codes this site expects, e.g. {"GJ", "HR"}. A
    reading whose state code is outside it pays state_penalty, so a code that
    is real but far away loses to a local one that is a single confusion away.
    That is what turns AR26BT1607 into HR26BT1607: AR is a genuine code, so
    nothing about the structure is wrong, but A and H look alike and this site
    sees HR daily while AR is 2,000km distant.

    The cost of that: a genuine Arunachal vehicle would be reported as HR. Set
    states to None (the default) to disable the preference entirely, or list
    every code the site really sees.
    """
    if not raw:
        return None

    text = re.sub(r"[^A-Z0-9]", "", raw.upper())

    if len(text) < MIN_LEN:
        return None

    best = None

    for d1 in (2, 1):            # 2-digit districts are far commoner
        for s in (2, 3, 1):      # 2-letter series likewise
            size = 2 + d1 + s + 4

            if size > len(text) or size > MAX_LEN:
                continue

            for start in range(0, len(text) - size + 1):
                chunk = text[start:start + size]

                district, c1 = _coerce(
                    chunk[2:2 + d1], "digit", max_repairs)
                if district is None:
                    continue

                series, c2 = _coerce(
                    chunk[2 + d1:2 + d1 + s], "alpha", max_repairs - c1)
                if series is None:
                    continue

                number, c3 = _coerce(
                    chunk[2 + d1 + s:], "digit", max_repairs - c1 - c2)
                if number is None:
                    continue

                for state, c0 in _state_options(
                        chunk[:2], states, state_penalty):
                    repairs = c0 + c1 + c2 + c3

                    if repairs > max_repairs:
                        break      # sorted, so the rest are worse

                    dropped = len(text) - size

                    # Discarding a character costs the same as repairing one.
                    # At 0.25 it was cheaper to drop a real digit than to fix a
                    # letter, so HR26B11607 lost its trailing 7 instead of
                    # having its 1 read as the T it was. The tie-break then
                    # prefers the reading that keeps every character.
                    score = (repairs + dropped, dropped)

                    if best is None or score < best[0]:
                        best = (score, {
                            "plate": state + district + series + number,
                            "state": state,
                            "district": district,
                            "series": series,
                            "number": number,
                            "repairs": round(repairs, 2),
                            "dropped": dropped,
                            "exact": repairs == 0 and dropped == 0,
                        })
                    break          # cheapest state for this segmentation

    return best[1] if best else None


def looks_like_plate(raw):
    """True if the string is already well formed, needing no repair at all."""
    m = PLATE_RE.match(re.sub(r"[^A-Z0-9]", "", (raw or "").upper()))
    return bool(m and m.group(1) in STATE_CODES)
