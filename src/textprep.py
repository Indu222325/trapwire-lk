"""Text normalisation and language detection shared by training and serving."""
import re
import unicodedata

URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(example|lk|com|net|xyz|top)\S*)", re.I)
PHONE_RE = re.compile(r"\b0\d{2}[-\s]?[x\d]{3}[-\s]?\d{3,4}\b", re.I)
MONEY_RE = re.compile(r"\b(rs\.?|rupees|රු\.?|ரூ\.?)\s*[\d,]+", re.I)
DIGIT_RE = re.compile(r"\d")
LEET_MAP = str.maketrans({"@": "a", "0": "o", "1": "i", "3": "e", "$": "s", "4": "a"})
SPACED_RE = re.compile(r"\b(?:[A-Za-z@$0-9] ){2,}[A-Za-z@$0-9]\b")
# Sinhala and Tamil vowel signs are combining marks that \w does not match,
# so the script ranges and the zero-width joiner are added explicitly.
INDIC = "\u0D80-\u0DFF\u0B80-\u0BFF\u200c\u200d"
TOKEN_RE = re.compile(rf"[^\W_]+(?:[{INDIC}\w]*)|[{INDIC}]+", re.UNICODE)
TOKEN_PATTERN = rf"(?u)[\w{INDIC}]{{2,}}"

SINGLISH_HINTS = {"oyage", "oya", "eka", "karanna", "ewanna", "nathnam", "salli", "wela", "machan",
                  "ada", "ganna", "danna", "kiyala", "thiyenne", "nemei", "amma", "mama", "ba", "ekata"}
TANGLISH_HINTS = {"unga", "ungalukku", "panna", "pannunga", "anupunga", "irukku", "aagiduchu", "udane",
                  "neenga", "naan", "innaiku", "naalaikku", "mattum", "kattunga", "podunga", "panam"}


def detect_lang(text: str) -> str:
    si = sum(1 for c in text if "඀" <= c <= "෿")
    ta = sum(1 for c in text if "஀" <= c <= "௿")
    if si or ta:
        return "si" if si >= ta else "ta"
    toks = set(t.lower() for t in TOKEN_RE.findall(text))
    s, t = len(toks & SINGLISH_HINTS), len(toks & TANGLISH_HINTS)
    if s == 0 and t == 0:
        return "en"
    return "singlish" if s >= t else "tanglish"


def normalize(text: str) -> str:
    """Replace volatile tokens with placeholders and undo common obfuscation."""
    t = unicodedata.normalize("NFC", str(text))
    t = URL_RE.sub(" URLTOKEN ", t)
    t = PHONE_RE.sub(" PHONETOKEN ", t)
    t = MONEY_RE.sub(" MONEYTOKEN ", t)
    # join spaced-out words such as "b l o c k e d"
    t = SPACED_RE.sub(lambda m: m.group(0).replace(" ", ""), t)
    # undo leetspeak only inside ASCII words that mix letters and symbols
    t = " ".join(w.translate(LEET_MAP) if w.isascii() and re.search(r"[a-zA-Z]", w) else w
                 for w in t.split())
    t = DIGIT_RE.sub("0", t)
    t = re.sub(rf"[^\w\s{INDIC}]", " ", t)  # drop emoji and punctuation, keep Sinhala/Tamil marks
    return re.sub(r"\s+", " ", t).strip().lower()


def tokens(text: str):
    return TOKEN_RE.findall(normalize(text))
