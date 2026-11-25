"""Shared utilities: language table, chat templating, token classification, prompt I/O."""
import json
import os
import pickle
import random
import unicodedata

import regex

ROOT = os.path.abspath(os.environ.get("WORLDDRAFT_ROOT") or os.environ.get("LEXDRAFT_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# code -> (name used in prompts, script tag, aya language name or None, mgsm code or None)
LANGS = {
    "en": ("English", "Latn", "English", "en"),
    "de": ("German", "Latn", "German", "de"),
    "fr": ("French", "Latn", "French", "fr"),
    "es": ("Spanish", "Latn", "Spanish", "es"),
    "sw": ("Swahili", "Latn", "Swahili", "sw"),
    "ru": ("Russian", "Cyrl", "Russian", "ru"),
    "zh": ("Simplified Chinese", "Hani", "Simplified Chinese", "zh"),
    "ja": ("Japanese", "Jpan", "Japanese", "ja"),
    "th": ("Thai", "Thai", "Thai", "th"),
    "bn": ("Bengali", "Beng", "Bengali", "bn"),
    "te": ("Telugu", "Telu", "Telugu", "te"),
    "ar": ("Arabic", "Arab", "Standard Arabic", None),
    "hi": ("Hindi", "Deva", "Hindi", None),
    "am": ("Amharic", "Ethi", "Amharic", None),
    "ko": ("Korean", "Hang", "Korean", None),
    "sr-Cyrl": ("Serbian, written in the Cyrillic script", "Cyrl", None, None),
    "sr-Latn": ("Serbian, written in the Latin script", "Latn", None, None),
    "hi-Latn": ("Hindi, written in romanized Latin script (not Devanagari)", "Latn", None, None),
}
NO_SPACE = {"zh", "ja", "th"}
MAX_NEW = {"math": 512, "translate": 320, "chat": 512}


def jload(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def jdump(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def pdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    os.replace(tmp, path)


def pload(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def shard(rows, i, n):
    return [r for k, r in enumerate(rows) if k % n == i]


def build_messages(task, lang, text):
    name = LANGS[lang][0]
    if task == "math":
        return [{"role": "user", "content": f"{text}\nPlease reason step by step in {name}, and put your final answer within \\boxed{{}}."}]
    if task == "translate":
        return [{"role": "user", "content": f"Translate the following text into {name}. Output only the translation.\n\n{text}"}]
    return [{"role": "user", "content": text}]


def chat_ids(tok, messages):
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=True)
    if isinstance(ids, dict) or hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return list(ids)


# --------------------------------------------------------------------------- tokens -> bytes

def token_byte_table(tok):
    """bytes for every token id of a byte-level BPE tokenizer (Qwen / Llama-3)."""
    try:
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
    except ImportError:  # newer transformers: standard GPT-2 byte <-> unicode map
        def bytes_to_unicode():
            bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
            cs = bs[:]
            n = 0
            for b_ in range(256):
                if b_ not in bs:
                    bs.append(b_)
                    cs.append(256 + n)
                    n += 1
            return dict(zip(bs, [chr(c) for c in cs]))
    dec = {v: k for k, v in bytes_to_unicode().items()}
    n = len(tok)
    table = []
    special = set(tok.all_special_ids)
    for i in range(n):
        s = tok.convert_ids_to_tokens(i)
        b = None
        if s is not None and i not in special:
            try:
                b = bytes(dec[c] for c in s)
            except KeyError:
                b = None
        if b is None:
            try:
                b = tok.decode([i]).encode("utf-8")
            except Exception:
                b = b""
        table.append(b)
    return table


def utf8_chars(buf):
    """Split bytes into (byte_start, byte_end, char); invalid bytes become U+FFFD of length 1."""
    out = []
    i, n = 0, len(buf)
    while i < n:
        b = buf[i]
        ln = 1 if b < 0x80 else 2 if 0xC0 <= b < 0xE0 else 3 if 0xE0 <= b < 0xF0 else 4 if 0xF0 <= b < 0xF8 else 0
        ch = None
        if ln and i + ln <= n:
            try:
                ch = buf[i:i + ln].decode("utf-8")
            except UnicodeDecodeError:
                ch = None
        if ch is None:
            out.append((i, i + 1, "�"))
            i += 1
        else:
            out.append((i, i + ln, ch))
            i += ln
    return out


_WORD_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+")
_SEG = {}


def _segment(text, lang):
    """Return list of (char_start, char_end) word spans."""
    if lang not in NO_SPACE:
        return [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]
    spans = []
    if lang == "zh":
        if "zh" not in _SEG:
            import jieba
            jieba.setLogLevel(60)
            _SEG["zh"] = jieba
        pieces = _SEG["zh"].cut(text, HMM=True)
    elif lang == "ja":
        if "ja" not in _SEG:
            import fugashi
            _SEG["ja"] = fugashi.Tagger()
        pieces = []
        pos = 0
        for w in _SEG["ja"](text):
            s = w.surface
            k = text.find(s, pos)
            if k < 0:
                continue
            if k > pos:
                pieces.append(text[pos:k])
            pieces.append(s)
            pos = k + len(s)
        if pos < len(text):
            pieces.append(text[pos:])
    else:
        if "th" not in _SEG:
            from pythainlp.tokenize import word_tokenize
            _SEG["th"] = word_tokenize
        pieces = _SEG["th"](text, engine="newmm", keep_whitespace=True)
    pos = 0
    for p in pieces:
        e = pos + len(p)
        # sub-split mixed pieces on non-word characters so punctuation is never part of a word
        for m in _WORD_RE.finditer(p):
            spans.append((pos + m.start(), pos + m.end()))
        pos = e
    return spans


def classify_tokens(ids, btable, lang):
    """Per-token word structure for a response.

    Returns dict of lists (len = len(ids)):
      cls: 0 other (whitespace/punct only), 1 word-start, 2 in-word
      wstart: token index of the word's start token (-1 if cls 0)
      j: index within word (0 for start)
      wlen: number of tokens in the word (0 if cls 0)
      bpiece: 1 if the token starts or ends inside a multi-byte character
    """
    starts, buf = [], bytearray()
    for t in ids:
        starts.append(len(buf))
        buf += btable[t]
    ends = starts[1:] + [len(buf)]
    chars = utf8_chars(bytes(buf))
    text = "".join(c for _, _, c in chars)
    cstart = [c[0] for c in chars]
    cend = [c[1] for c in chars]
    charbound = set(cstart) | {len(buf)}
    words = _segment(text, lang)
    n = len(ids)
    cls, wst, jj, wl = [0] * n, [-1] * n, [0] * n, [0] * n
    bpiece = [int(starts[k] not in charbound or ends[k] not in charbound) for k in range(n)]
    # token index for each byte offset
    import bisect
    for (a, b) in words:
        if a >= len(cstart):
            continue
        wb0, wb1 = cstart[a], cend[b - 1]
        k0 = bisect.bisect_right(starts, wb0) - 1
        k1 = bisect.bisect_left(starts, wb1)  # tokens with start < wb1
        toks = list(range(max(k0, 0), k1))
        if not toks:
            continue
        if cls[toks[0]] == 0:
            cls[toks[0]], wst[toks[0]], jj[toks[0]] = 1, toks[0], 0
        for idx, k in enumerate(toks[1:], 1):
            if cls[k] == 0:
                cls[k], wst[k], jj[k] = 2, toks[0], idx
        for k in toks:
            if wst[k] == toks[0]:
                wl[k] = len(toks)
    return {"cls": cls, "wstart": wst, "j": jj, "wlen": wl, "bpiece": bpiece}


SCRIPT_RANGES = {
    "Latn": [(0x41, 0x24F)], "Cyrl": [(0x400, 0x52F)], "Hani": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
    "Jpan": [(0x3040, 0x30FF), (0x4E00, 0x9FFF)], "Thai": [(0xE00, 0xE7F)], "Beng": [(0x980, 0x9FF)],
    "Telu": [(0xC00, 0xC7F)], "Arab": [(0x600, 0x6FF), (0x750, 0x77F)], "Deva": [(0x900, 0x97F)],
    "Ethi": [(0x1200, 0x139F)], "Hang": [(0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)],
}


def script_ratio(text, script):
    letters = [c for c in text if unicodedata.category(c)[0] == "L"]
    if not letters:
        return 0.0
    rg = SCRIPT_RANGES[script]
    hit = sum(1 for c in letters if any(a <= ord(c) <= b for a, b in rg))
    return hit / len(letters)


def lang_ok(text, lang):
    """Coarse check that the response is in the requested language/script."""
    script = LANGS[lang][1]
    r = script_ratio(text, script)
    if script != "Latn":
        if lang == "ja":
            kana = script_ratio(text, "Jpan") - script_ratio(text, "Hani") * 0  # kana or kanji
            has_kana = any(0x3040 <= ord(c) <= 0x30FF for c in text)
            return r > 0.5 and has_kana
        if lang == "zh":
            return r > 0.5 and not any(0x3040 <= ord(c) <= 0x30FF for c in text)
        return r > 0.5
    if r < 0.8:
        return False
    try:
        import langid
    except ImportError:
        return True
    code = {"en": "en", "de": "de", "fr": "fr", "es": "es", "sw": "sw", "sr-Latn": "sr", "hi-Latn": None}[lang]
    if code is None:
        return True
    if lang == "sr-Latn":
        return langid.classify(text)[0] in ("sr", "hr", "bs", "sl")
    return langid.classify(text)[0] == code


def set_seed(s=0):
    random.seed(s)
