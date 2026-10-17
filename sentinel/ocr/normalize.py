"""Vietnamese plate text: normalise, repair look-alike characters by position, validate the format.

Formats (separators removed):
    car / truck     30A12345            2 digits (province) + 1 letter + 4-5 digits
    motorcycle      29B112345, 29B11234 2 digits + 1 letter + 1 digit + 4-5 digits
    two-letter      29MD112345          2 digits + 2 letters + (1 digit) + 4-5 digits
This only checks the SHAPE of the text; it says nothing about whether the plate exists or whether the reading is right.
"""
import re

# series letters actually used on Vietnamese plates: I, J, O, Q, R, W are never issued
LETTERS = "ABCDEFGHKLMNPSTUVXYZ"
PLATE_RE = re.compile(r"^\d{2}[" + LETTERS + r"]{1,2}\d?\d{4,5}$")

# characters an OCR engine confuses, applied only where the position demands the other class
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "T": "1", "Z": "2", "A": "4", "S": "5", "G": "6", "B": "8"}
TO_LETTER = {"0": "D", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B"}

# position templates (N = digit, L = letter), tried in order; the cheapest valid repair wins
TEMPLATES = ["NNLNNNNN", "NNLNNNNNN", "NNLLNNNNN", "NNLLNNNNNN", "NNLNNNN", "NNLLNNNN"]


def clean(raw):
    """'29-b1 123.45' -> '29B112345' (upper case, letters and digits only)."""
    return re.sub(r"[^0-9A-Z]", "", str(raw or "").upper())


def is_valid(text):
    return bool(text) and PLATE_RE.match(text) is not None


def repair(text):
    """Cheapest repair of look-alike characters that yields a valid plate -> (text, n_substitutions) or (None, None)."""
    best = (None, None)
    for t in TEMPLATES:
        if len(t) != len(text):
            continue
        out, cost, ok = [], 0, True
        for ch, kind in zip(text, t):
            if kind == "N":
                if ch.isdigit():
                    out.append(ch)
                elif ch in TO_DIGIT:
                    out.append(TO_DIGIT[ch]); cost += 1
                else:
                    ok = False; break
            else:
                if ch in LETTERS:
                    out.append(ch)
                elif ch in TO_LETTER:
                    out.append(TO_LETTER[ch]); cost += 1
                else:
                    ok = False; break
        cand = "".join(out)
        if ok and is_valid(cand) and (best[1] is None or cost < best[1]):
            best = (cand, cost)
    return best


def normalize(raw):
    """-> {"text": cleaned/repaired string or None, "clean": cleaned raw, "format_ok": bool, "repairs": int}.
    A reading that cannot be repaired into the plate shape gives text=None (the raw characters stay in "clean")."""
    c = clean(raw)
    if is_valid(c):
        return {"text": c, "clean": c, "format_ok": True, "repairs": 0}
    fixed, cost = repair(c)
    if fixed:
        return {"text": fixed, "clean": c, "format_ok": True, "repairs": cost}
    return {"text": None, "clean": c, "format_ok": False, "repairs": 0}
