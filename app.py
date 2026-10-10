# %% [markdown] Header
"""
SafeSignal - NLP toolkit for spotting hidden drug side-effects in patient posts.

Course : Natural Language Processing (0304220301), Symbiosis AI Institute
Author : Sanobar Latif Shaikh (PRN 25030421039)

Run locally :  python app.py
Run in Colab:  open notebooks/SafeSignal_Colab.ipynb and press "Run all"
"""

# %% Imports
import os
import re
import sys
import math
import random
import inspect
import subprocess
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
    ENGLISH_STOP_WORDS,
)
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

try:  # spaCy is optional: the app still runs (rule-based) if it is missing
    import spacy
    HAS_SPACY = True
except Exception:  # pragma: no cover
    spacy = None
    HAS_SPACY = False

APP_TITLE = "SafeSignal"
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
MODEL_DIR = os.environ.get("SAFESIGNAL_MODEL", os.path.join(BASE_DIR, "models", "safesignal_cadec"))

LABEL_COLORS = {
    "DRUG": "#6C8EF5",
    "ADR": "#FF7A59",
    "DISEASE": "#F5B83D",
    "SYMPTOM": "#4CC9A6",
    "FINDING": "#B28DFF",
}

# %% Lexicons and sample data
DRUGS = sorted(set("""
lipitor atorvastatin voltaren diclofenac arthrotec zocor simvastatin crestor rosuvastatin
ibuprofen naproxen aspirin paracetamol acetaminophen tylenol metformin lisinopril amlodipine
omeprazole sertraline prozac zoloft cymbalta meloxicam flexeril cataflam pennsaid zipsor
gabapentin prednisone amoxicillin azithromycin levothyroxine warfarin plavix xarelto losartan
metoprolol celebrex advil motrin aleve
""".split()))
DRUG_SET = set(DRUGS)

ADR_TERMS = [
    "muscle pain", "muscle aches", "muscle weakness", "joint pain", "memory loss", "memory problems",
    "memory problem", "nausea", "vomiting", "headache", "dizziness", "dizzy", "fatigue", "tiredness",
    "insomnia", "can't sleep", "cannot sleep", "trouble sleeping", "stomach pain", "stomach upset",
    "diarrhea", "constipation", "rash", "itching", "swelling", "chest pain", "blurred vision",
    "weight gain", "hair loss", "depression", "anxiety", "mood swings", "cramp", "leg cramp",
    "back pain", "shortness of breath", "dry mouth", "loss of appetite", "heartburn", "confusion",
    "brain fog", "tremors", "sweating", "palpitations", "numbness", "tingling", "drowsiness",
    "liver problems", "stomach ache", "pain in my legs", "aching joints",
]
DISEASE_TERMS = [
    "diabetes", "arthritis", "osteoarthritis", "rheumatoid arthritis", "hypertension", "high cholesterol",
    "high blood pressure", "asthma", "migraine", "heart disease", "epilepsy", "thyroid", "gerd",
]
SYMPTOM_TERMS = ["fever", "cough", "stiffness", "inflammation", "swollen joints", "sore throat", "chills"]
FINDING_TERMS = [
    "elevated liver enzymes", "high blood sugar", "abnormal blood test", "low platelet count",
    "raised ldl", "high ldl", "elevated cpk", "blood test",
]
MODIFIERS = r"(?:severe|terrible|bad|extreme|constant|mild|awful|horrible|intense|unbearable|sharp|chronic|slight|persistent)"
SEVERE_WORDS = {"severe", "terrible", "unbearable", "extreme", "awful", "horrible", "intense", "debilitating", "worst"}
STOP_WORDS_SIGNAL = {"stopped", "quit", "discontinued", "switched", "stop"}

SAMPLE_POSTS = [
    "I took Lipitor and started having severe muscle pain and memory problems.",
    "Been on Voltaren gel for my arthritis for 3 weeks. Terrible stomach pain and nausea every morning.",
    "Started Zocor for high cholesterol. After two weeks I got constant leg cramps and can't sleep at night.",
    "Crestor worked on my cholesterol but the muscle weakness and fatigue made me stop it.",
    "lipiter gave me bad headaches n dizzy all day, doc switched me to something else",
    "Arthrotec helped my osteoarthritis but I get diarrhea and stomach upset almost daily.",
    "My doctor put me on metformin for diabetes. Nausea and diarrhea for the first week, then fine.",
    "Took ibuprofen for back pain, got heartburn and a rash on my arms.",
    "Sertraline made me feel numb, sweating at night and I had trouble sleeping. Quit after a month.",
    "Atorvastatin 20 mg: severe joint pain, brain fog, and my blood test showed elevated liver enzymes.",
    "Naproxen for my migraine. Dizziness and dry mouth, otherwise okay.",
    "Zocor gave me terrible memory loss, I kept forgetting simple words. Stopped it last week.",
    "Prednisone for asthma caused weight gain, mood swings and insomnia.",
    "voltarin tablets = awful stomach ache and swelling in my ankles",
    "Lipitor for 2 years, now I have muscle pain and can't sleep most nights.",
    "Crestor 10mg, no problems for months, then sudden chest pain and shortness of breath. Scary.",
    "Been taking amlodipine for high blood pressure, swelling in my feet and constant tiredness.",
    "Diclofenac helped my back pain but caused persistent heartburn and nausea.",
    "Gabapentin gives me drowsiness and confusion, I feel like a zombie all day.",
    "Started omeprazole for GERD, now I get headaches and constipation.",
    "Simvastatin made my legs ache so bad I could barely climb stairs. Muscle pain is unbearable.",
    "Celebrex is great for arthritis but I noticed swelling and a rash after a week.",
    "Lisinopril cough is real! dry cough every night and some dizziness.",
    "I switched from Zocor to Crestor, still getting muscle aches and tiredness though.",
    "Meloxicam works fine for my joint pain, only mild nausea sometimes.",
    "Zoloft: anxiety went down but I got tremors and insomnia for the first two weeks.",
    "Taking Arthrotec with food helps but I still get severe stomach pain.",
    "Pennsaid on my knee gave me itching and a nasty rash.",
]

# %% Optional NLP resources (NLTK) with graceful fallbacks
def _load_wordnet():
    try:
        import nltk
        from nltk.corpus import wordnet
        try:
            wordnet.synsets("pain")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            wordnet.synsets("pain")
        return wordnet
    except Exception:
        return None


def _load_vader():
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        try:
            return SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            return SentimentIntensityAnalyzer()
    except Exception:
        return None


FALLBACK_LEXICON = {
    "good": 1.9, "great": 3.1, "better": 1.9, "helped": 1.8, "works": 1.4, "fine": 1.1, "love": 3.2,
    "relief": 2.0, "happy": 2.7, "okay": 0.9, "worked": 1.5, "bad": -2.5, "terrible": -2.1,
    "awful": -2.0, "horrible": -2.5, "worst": -3.1, "pain": -2.3, "nausea": -1.9, "sick": -2.1,
    "scary": -2.2, "unbearable": -2.7, "severe": -2.0, "stopped": -1.0, "quit": -1.1, "sleep": 0.0,
    "headache": -1.6, "headaches": -1.6, "dizzy": -1.5, "tired": -1.4, "rash": -1.6, "swelling": -1.4,
    "cramps": -1.7, "diarrhea": -1.8, "weak": -1.6, "zombie": -1.8, "confusion": -1.7,
}

_RESOURCES = {}


def get_resource(name):
    """Lazy loader so that the app starts quickly and never crashes on a missing corpus."""
    if name not in _RESOURCES:
        _RESOURCES[name] = {"wordnet": _load_wordnet, "vader": _load_vader}[name]()
    return _RESOURCES[name]


# %% Tokenisation helpers and edit distance (Unit 2: word-level analysis)
_TOKENIZER = None


def tokenize(text):
    """spaCy tokenizer when available, otherwise a regex tokenizer."""
    global _TOKENIZER
    if HAS_SPACY:
        try:
            if _TOKENIZER is None:
                _TOKENIZER = spacy.blank("en")
            return [t.text for t in _TOKENIZER(text) if not t.is_space]
        except Exception:
            pass
    return re.findall(r"[A-Za-z0-9]+(?:'[a-z]+)?|[^\sA-Za-z0-9]", text)


def levenshtein(a, b):
    a, b = a.lower(), b.lower()
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def levenshtein_matrix(a, b):
    a, b = a.lower(), b.lower()
    m = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        m[i][0] = i
    for j in range(len(b) + 1):
        m[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            m[i][j] = min(m[i - 1][j] + 1, m[i][j - 1] + 1, m[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    cols = ["-"] + list(b)
    rows = ["-"] + list(a)
    return pd.DataFrame(m, index=rows, columns=cols)


def closest_drugs(word, top=6):
    word = word.strip().lower()
    if not word:
        return pd.DataFrame(columns=["Drug", "Edit distance", "Similarity %"])
    rows = []
    for d in DRUGS:
        dist = levenshtein(word, d)
        sim = round(100 * (1 - dist / max(len(word), len(d))), 1)
        rows.append((d.title(), dist, sim))
    rows.sort(key=lambda r: (r[1], r[0]))
    return pd.DataFrame(rows[:top], columns=["Drug", "Edit distance", "Similarity %"])


# %% Entity extraction engine (rules + fuzzy matching + optional trained spaCy NER)
def _pat(phrase):
    p = re.escape(phrase).replace("\\ ", r"\s+").replace("'", "['\u2019]?")
    return p


def _compile(words, prefix="", plural=True):
    alts = "|".join(sorted((_pat(w) for w in words), key=len, reverse=True))
    tail = "s?" if plural else ""
    return re.compile(rf"(?<![A-Za-z])(?:{prefix}(?:{alts}){tail})(?![A-Za-z])", re.I)


RULES = {
    "DRUG": _compile(DRUGS, plural=False),
    "ADR": _compile(ADR_TERMS, prefix=rf"(?:{MODIFIERS}\s+)?"),
    "DISEASE": _compile(DISEASE_TERMS),
    "SYMPTOM": _compile(SYMPTOM_TERMS),
    "FINDING": _compile(FINDING_TERMS),
}


def _ent(start, end, text, label, source, canon=None):
    return {"start": start, "end": end, "text": text, "label": label, "source": source, "canonical": canon}


def fuzzy_drug_entities(text, taken):
    """Catch misspelt drug names such as 'lipiter' or 'voltarin' using edit distance."""
    out = []
    for m in re.finditer(r"[A-Za-z]{6,}", text):
        w = m.group().lower()
        if w in DRUG_SET or w in ENGLISH_STOP_WORDS:
            continue
        if any(m.start() < e and s < m.end() for s, e in taken):
            continue
        limit = 1 if len(w) <= 7 else 2
        best, best_d = None, 99
        for d in DRUGS:
            if d[0] != w[0] or abs(len(d) - len(w)) > 2:
                continue
            dist = levenshtein(w, d)
            if dist < best_d:
                best, best_d = d, dist
        if best is not None and best_d <= limit:
            out.append(_ent(m.start(), m.end(), m.group(), "DRUG", "fuzzy", best))
    return out


def rule_entities(text):
    ents = []
    for label, rx in RULES.items():
        for m in rx.finditer(text):
            ents.append(_ent(m.start(), m.end(), m.group(), label, "rule"))
    taken = [(e["start"], e["end"]) for e in ents]
    ents.extend(fuzzy_drug_entities(text, taken))
    return ents


def resolve_overlaps(ents):
    """Keep the longest span, prefer the trained model when lengths tie."""
    ordered = sorted(ents, key=lambda e: (-(e["end"] - e["start"]), e["source"] != "model", e["start"]))
    kept = []
    for e in ordered:
        if all(e["end"] <= k["start"] or e["start"] >= k["end"] for k in kept):
            kept.append(e)
    return sorted(kept, key=lambda e: e["start"])


# --- demo model: trained on synthetic template data so the app is fully self-contained
def _drug_typo(d, rng):
    if len(d) > 5 and rng.random() < 0.5:
        i = rng.randrange(1, len(d) - 1)
        return d[:i] + d[i + 1:]
    return d


def make_silver_data(n=450, seed=7):
    rng = random.Random(seed)
    adrs = [a for a in ADR_TERMS if "'" not in a]
    templates = [
        ["I took ", "D", " and started having ", "A", " and ", "A", "."],
        ["Been on ", "D", " for 3 weeks. ", "M", "A", " every morning."],
        ["My ", "DIS", " got better but ", "D", " caused ", "A", "."],
        ["Doctor put me on ", "D", " for my ", "DIS", ". Now I get ", "A", " every day."],
        ["Stopped ", "D", " because of ", "M", "A", ". Also noticed ", "S", "."],
        ["Lab report showed ", "F", " after starting ", "D", "."],
        ["i hav ", "A", " since startin ", "D", " lol"],
        ["After ", "D", " I got ", "A", ", ", "A", " and ", "A", "."],
        ["Switched from ", "D", " to ", "D", ", still getting ", "A", "."],
        ["", "D", " is okay for ", "DIS", " but the ", "A", " is hard to ignore."],
    ]
    pools = {"D": DRUGS, "A": adrs, "DIS": DISEASE_TERMS, "S": SYMPTOM_TERMS, "F": FINDING_TERMS}
    labels = {"D": "DRUG", "A": "ADR", "DIS": "DISEASE", "S": "SYMPTOM", "F": "FINDING"}
    mods = ["", "", "severe ", "terrible ", "constant ", "bad "]
    data = []
    for _ in range(n):
        tpl = rng.choice(templates)
        text, ents, pending_mod = "", [], ""
        for seg in tpl:
            if seg == "M":
                pending_mod = rng.choice(mods)
                continue
            if seg in pools:
                val = rng.choice(pools[seg])
                if seg == "D":
                    val = _drug_typo(val, rng)
                    if rng.random() < 0.3:
                        val = val.capitalize()
                if seg == "A":
                    val = pending_mod + val
                    pending_mod = ""
                ents.append((len(text), len(text) + len(val), labels[seg]))
                text += val
            else:
                text += seg
        data.append((text, ents))
    return data


def train_demo_model(epochs=12, seed=7):
    """Trains a small spaCy NER on synthetic data (about 10-20 seconds on CPU)."""
    if not HAS_SPACY:
        return None
    from spacy.training import Example
    from spacy.util import minibatch, compounding

    random.seed(seed)
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    for lab in LABEL_COLORS:
        ner.add_label(lab)
    train = make_silver_data(seed=seed)
    examples = []
    for text, ents in train:
        doc = nlp.make_doc(text)
        examples.append(Example.from_dict(doc, {"entities": ents}))
    optimizer = nlp.initialize(lambda: examples)
    for _ in range(epochs):
        random.shuffle(examples)
        losses = {}
        for batch in minibatch(examples, size=compounding(4.0, 24.0, 1.001)):
            nlp.update(batch, sgd=optimizer, drop=0.2, losses=losses)
    return nlp


_MODEL = {"nlp": None, "kind": None, "tried": False}


def get_model():
    """Loads the CADEC-trained model if present, otherwise trains the self-contained demo model."""
    if _MODEL["tried"]:
        return _MODEL["nlp"], _MODEL["kind"]
    _MODEL["tried"] = True
    print("SafeSignal: preparing the NER model (first run only, ~20 s)...")
    if HAS_SPACY and os.path.isdir(MODEL_DIR):
        try:
            _MODEL["nlp"], _MODEL["kind"] = spacy.load(MODEL_DIR), "CADEC-trained model"
            return _MODEL["nlp"], _MODEL["kind"]
        except Exception:
            pass
    try:
        _MODEL["nlp"], _MODEL["kind"] = train_demo_model(), "demo model (synthetic data)"
    except Exception:
        _MODEL["nlp"], _MODEL["kind"] = None, None
    return _MODEL["nlp"], _MODEL["kind"]


def model_entities(text):
    nlp, _ = get_model()
    if nlp is None:
        return []
    doc = nlp(text)
    return [_ent(e.start_char, e.end_char, e.text, e.label_.upper(), "model") for e in doc.ents if e.label_.upper() in LABEL_COLORS]


ENGINES = ["Hybrid (recommended)", "Rule-based", "Trained NER"]


def extract_entities(text, engine="Hybrid (recommended)"):
    text = text or ""
    ents = []
    if engine != "Rule-based":
        ents += model_entities(text)
    if engine != "Trained NER" or not ents:
        ents += rule_entities(text)
    return resolve_overlaps(ents)


def normalise(ent):
    if ent.get("canonical"):
        return ent["canonical"].lower()
    t = ent["text"].lower().strip()
    t = re.sub(rf"^{MODIFIERS}\s+", "", t)
    return re.sub(r"\s+", " ", t)


def to_segments(text, ents):
    segs, pos = [], 0
    for e in ents:
        if e["start"] > pos:
            segs.append((text[pos:e["start"]], None))
        segs.append((text[e["start"]:e["end"]], e["label"]))
        pos = e["end"]
    if pos < len(text):
        segs.append((text[pos:], None))
    return segs or [(text, None)]


def entity_table(ents):
    rows = [(e["text"], e["label"], normalise(e), e["source"]) for e in ents]
    return pd.DataFrame(rows, columns=["Mention", "Type", "Normalised", "Found by"])


def risk_assessment(text, ents):
    adrs = [e for e in ents if e["label"] == "ADR"]
    low = text.lower()
    severe = sum(1 for w in SEVERE_WORDS if re.search(rf"\b{w}\b", low))
    stopped = sum(1 for w in STOP_WORDS_SIGNAL if re.search(rf"\b{w}\b", low))
    score = min(100, 18 * len(adrs) + 12 * severe + 15 * stopped)
    level = "Low" if score < 30 else ("Moderate" if score < 60 else "High")
    return score, level, len(adrs), severe, stopped


def risk_card(text, ents):
    score, level, n_adr, severe, stopped = risk_assessment(text, ents)
    color = {"Low": "#2ecc71", "Moderate": "#f39c12", "High": "#e74c3c"}[level]
    drugs = sorted({normalise(e) for e in ents if e["label"] == "DRUG"})
    adrs = sorted({normalise(e) for e in ents if e["label"] == "ADR"})
    return f"""
<div style="border-radius:14px;padding:16px 18px;background:#101826;color:#eaf2ff;border-left:8px solid {color};font-family:sans-serif">
  <div style="font-size:13px;opacity:.7;letter-spacing:1px">SAFETY SIGNAL SCORE</div>
  <div style="font-size:34px;font-weight:700;color:{color}">{score}/100 &nbsp;<span style="font-size:18px">{level}</span></div>
  <div style="margin-top:8px;font-size:14px;line-height:1.7">
    <b>Drugs:</b> {', '.join(drugs) if drugs else 'none found'}<br>
    <b>Reactions:</b> {', '.join(adrs) if adrs else 'none found'}<br>
    <b>Severity words:</b> {severe} &nbsp;|&nbsp; <b>Stopped / switched drug:</b> {'yes' if stopped else 'no'}
  </div>
  <div style="margin-top:8px;font-size:12px;opacity:.6">Heuristic score for demonstration. Not medical advice.</div>
</div>"""


def analyze_post(text, engine):
    text = (text or "").strip()
    if not text:
        return [("Please enter a patient post.", None)], pd.DataFrame(), "<i>Waiting for input...</i>"
    ents = extract_entities(text, engine)
    return to_segments(text, ents), entity_table(ents), risk_card(text, ents)


def engine_status():
    _, kind = get_model()
    if kind is None:
        return "Trained NER unavailable (spaCy not installed). Rule-based engine is active."
    return f"Trained NER in use: **{kind}**. Put a CADEC-trained spaCy model in `models/safesignal_cadec` to use it automatically."


# %% Batch analysis: turning many posts into drug-reaction signals
def _read_posts(file_obj, pasted):
    posts = []
    if file_obj is not None:
        path = getattr(file_obj, "name", file_obj)
        try:
            if str(path).lower().endswith(".csv"):
                df = pd.read_csv(path)
                col = "text" if "text" in df.columns else df.columns[0]
                posts = [str(x) for x in df[col].dropna().tolist()]
            else:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    posts = [ln.strip() for ln in f if ln.strip()]
        except Exception:
            posts = []
    if not posts and pasted:
        posts = [ln.strip() for ln in pasted.splitlines() if ln.strip()]
    return posts


def _empty_fig(msg="Nothing to show yet"):
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12, color="#777")
    ax.axis("off")
    return fig


def batch_analyze(file_obj, pasted, engine):
    posts = _read_posts(file_obj, pasted)
    if not posts:
        return "Add posts (one per line) or upload a CSV with a `text` column.", pd.DataFrame(), pd.DataFrame(), _empty_fig(), None
    drug_c, adr_c, pair_c, rows = Counter(), Counter(), Counter(), []
    for i, p in enumerate(posts, 1):
        ents = extract_entities(p, engine)
        drugs = sorted({normalise(e) for e in ents if e["label"] == "DRUG"})
        adrs = sorted({normalise(e) for e in ents if e["label"] == "ADR"})
        drug_c.update(drugs)
        adr_c.update(adrs)
        for d in drugs:
            for a in adrs:
                pair_c[(d, a)] += 1
        rows.append((i, p[:90] + ("..." if len(p) > 90 else ""), ", ".join(drugs), ", ".join(adrs)))
    detail = pd.DataFrame(rows, columns=["#", "Post", "Drugs", "Reactions"])
    pairs = pd.DataFrame(
        [(d, a, c) for (d, a), c in pair_c.most_common(25)], columns=["Drug", "Reaction", "Posts reporting both"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, counter, title, color in (
        (axes[0], drug_c, "Most mentioned drugs", "#6C8EF5"),
        (axes[1], adr_c, "Most reported reactions", "#FF7A59"),
    ):
        items = counter.most_common(8)[::-1]
        if items:
            ax.barh([k for k, _ in items], [v for _, v in items], color=color)
        ax.set_title(title, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_path = "/tmp/safesignal_batch_results.csv"
    try:
        detail.to_csv(out_path, index=False)
    except Exception:
        out_path = None
    summary = (
        f"**{len(posts)} posts analysed** - {sum(drug_c.values())} drug mentions, "
        f"{sum(adr_c.values())} reaction mentions, {len(pair_c)} distinct drug-reaction pairs."
    )
    return summary, pairs, detail, fig, out_path


# %% Unit 1-2: preprocessing lab and regular expressions
def stem_word(w):
    try:
        from nltk.stem import PorterStemmer
        return PorterStemmer().stem(w)
    except Exception:
        for suf in ("ing", "ed", "es", "s"):
            if w.endswith(suf) and len(w) > len(suf) + 2:
                return w[: -len(suf)]
        return w


def lemmatize_word(w):
    wn = get_resource("wordnet")
    if wn is not None:
        try:
            from nltk.stem import WordNetLemmatizer
            lem = WordNetLemmatizer()
            out = lem.lemmatize(w, "v")
            return lem.lemmatize(out, "n")
        except Exception:
            pass
    return stem_word(w)


def preprocess_text(text, lower, punct, stop, stem, lemma):
    tokens = tokenize(text or "")
    rows = []
    final = []
    for tok in tokens:
        cur, note = tok, []
        if lower:
            cur = cur.lower()
        if punct and not re.search(r"[A-Za-z0-9]", cur):
            rows.append((tok, "", "punctuation removed"))
            continue
        if stop and cur.lower() in ENGLISH_STOP_WORDS:
            rows.append((tok, "", "stop-word removed"))
            continue
        if lemma:
            new = lemmatize_word(cur.lower())
            if new != cur.lower():
                note.append("lemmatised")
            cur = new
        if stem:
            new = stem_word(cur.lower())
            if new != cur.lower():
                note.append("stemmed")
            cur = new
        final.append(cur)
        rows.append((tok, cur, ", ".join(note)))
    df = pd.DataFrame(rows, columns=["Original token", "After processing", "Action"])
    info = f"**{len(tokens)} tokens in, {len(final)} tokens out.**  \nResult: `{' '.join(final)}`"
    return df, info


REGEX_PRESETS = {
    "Dosage (e.g. 20 mg)": r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|ml|g)\b",
    "Duration (e.g. 3 weeks)": r"\b\d+\s+(?:day|week|month|year)s?\b",
    "Hashtags": r"#\w+",
    "Email address": r"[\w.+-]+@[\w-]+\.[\w.-]+",
    "Repeated letters (sooo)": r"(\w)\1{2,}",
}


def regex_tester(text, preset, custom):
    pattern = (custom or "").strip() or REGEX_PRESETS.get(preset, "")
    if not pattern:
        return pd.DataFrame(), "Choose a preset or type a pattern."
    try:
        rx = re.compile(pattern, re.I)
    except re.error as e:
        return pd.DataFrame(), f"Invalid regular expression: {e}"
    rows = [(m.group(), m.start(), m.end()) for m in rx.finditer(text or "")]
    return pd.DataFrame(rows, columns=["Match", "Start", "End"]), f"`{pattern}` found **{len(rows)}** match(es)."


# %% Unit 2: n-gram language model with add-k smoothing
class NGramLM:
    def __init__(self, n=2, k=0.1):
        self.n, self.k = n, k
        self.ngrams, self.ctx, self.vocab = Counter(), Counter(), set()

    @staticmethod
    def words(sentence):
        return re.findall(r"[a-z]+(?:'[a-z]+)?", sentence.lower())

    def fit(self, corpus):
        sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", corpus) if s.strip()]
        for s in sentences:
            toks = ["<s>"] * (self.n - 1) + self.words(s) + ["</s>"]
            self.vocab.update(toks)
            for i in range(self.n - 1, len(toks)):
                ctx = tuple(toks[i - self.n + 1:i])
                self.ngrams[ctx + (toks[i],)] += 1
                self.ctx[ctx] += 1
        self.vocab.discard("<s>")
        return self

    def prob(self, ctx, w):
        v = len(self.vocab) + 1
        return (self.ngrams[ctx + (w,)] + self.k) / (self.ctx[ctx] + self.k * v)

    def context_of(self, text):
        toks = ["<s>"] * (self.n - 1) + self.words(text)
        return tuple(toks[len(toks) - (self.n - 1):]) if self.n > 1 else ()

    def predict(self, text, top=8):
        ctx = self.context_of(text)
        scores = [(w, self.prob(ctx, w)) for w in self.vocab]
        scores.sort(key=lambda x: -x[1])
        return scores[:top]

    def perplexity(self, sentence):
        toks = ["<s>"] * (self.n - 1) + self.words(sentence) + ["</s>"]
        logp, cnt = 0.0, 0
        for i in range(self.n - 1, len(toks)):
            ctx = tuple(toks[i - self.n + 1:i])
            logp += math.log(self.prob(ctx, toks[i]))
            cnt += 1
        return math.exp(-logp / max(cnt, 1))

    def generate(self, seed_text, length=12, rng=None):
        rng = rng or random.Random(1)
        out = self.words(seed_text)
        for _ in range(length):
            cand = [(w, p) for w, p in self.predict(" ".join(out), top=6)]
            words, probs = zip(*cand)
            w = rng.choices(words, weights=probs)[0]
            if w == "</s>":
                break
            out.append(w)
        return " ".join(out)


def ngram_demo(corpus, order, k, seed_text, test_sentence):
    corpus = (corpus or "").strip() or "\n".join(SAMPLE_POSTS)
    n = {"Unigram": 1, "Bigram": 2, "Trigram": 3}[order]
    lm = NGramLM(n, float(k)).fit(corpus)
    preds = lm.predict(seed_text or "", top=8)
    df = pd.DataFrame([(w, round(p, 4)) for w, p in preds], columns=["Next word", "Probability"])
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh([w for w, _ in preds][::-1], [p for _, p in preds][::-1], color="#1abc9c")
    ax.set_title(f"{order} model: next-word probabilities", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    ppl = lm.perplexity(test_sentence) if (test_sentence or "").strip() else float("nan")
    gen = lm.generate(seed_text or "i took")
    info = f"Vocabulary: **{len(lm.vocab)}** words  |  Perplexity of test sentence: **{ppl:.1f}**  \nGenerated text: *{gen}*"
    return df, fig, info


# %% Unit 3: feature representations and embeddings
def _docs(text):
    d = [x.strip() for x in (text or "").splitlines() if x.strip()]
    return d if d else list(SAMPLE_POSTS[:6])


def feature_demo(text, method, use_bigrams, drop_stop):
    docs = _docs(text)
    ngram = (1, 2) if use_bigrams else (1, 1)
    stop = "english" if drop_stop else None
    try:
        if method.startswith("TF-IDF"):
            vec = TfidfVectorizer(ngram_range=ngram, stop_words=stop)
        elif method.startswith("One-Hot"):
            vec = CountVectorizer(ngram_range=ngram, stop_words=stop, binary=True)
        else:
            vec = CountVectorizer(ngram_range=ngram, stop_words=stop)
        X = vec.fit_transform(docs)
    except ValueError:
        return pd.DataFrame(), _empty_fig("Vocabulary is empty. Try different text."), "No usable words found."
    terms = vec.get_feature_names_out()
    dense = X.toarray()
    order = np.argsort(-dense.sum(axis=0))[:20]
    df = pd.DataFrame(dense[:, order].round(3), columns=[terms[i] for i in order])
    df.insert(0, "Doc", [f"D{i + 1}" for i in range(len(docs))])
    fig, ax = plt.subplots(figsize=(8.5, max(2.5, 0.45 * len(docs) + 1.5)))
    im = ax.imshow(dense[:, order], aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([terms[i] for i in order], rotation=70, ha="right", fontsize=8)
    ax.set_yticks(range(len(docs)))
    ax.set_yticklabels([f"D{i + 1}" for i in range(len(docs))], fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title(f"{method} (top {len(order)} terms)", fontweight="bold")
    fig.tight_layout()
    info = f"Matrix shape: **{dense.shape[0]} documents x {dense.shape[1]} features**, sparsity {100 * (1 - np.count_nonzero(dense) / dense.size):.1f}%"
    return df, fig, info


def embedding_demo(text, query_text, query_word):
    docs = _docs(text) if (text or "").strip() else list(SAMPLE_POSTS)
    try:
        vec = TfidfVectorizer(stop_words="english")
        X = vec.fit_transform(docs)
    except ValueError:
        return _empty_fig("Not enough text"), pd.DataFrame(), pd.DataFrame(), "Add more text."
    k = min(10, X.shape[0] - 1, X.shape[1] - 1)
    if k < 2:
        return _empty_fig("Need at least 3 documents"), pd.DataFrame(), pd.DataFrame(), "Add more documents (one per line)."
    svd = TruncatedSVD(n_components=k, random_state=0)
    Z = svd.fit_transform(X)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].scatter(Z[:, 0], Z[:, 1], s=70, c=np.arange(len(docs)), cmap="plasma")
    for i in range(len(docs)):
        axes[0].annotate(f"D{i + 1}", (Z[i, 0], Z[i, 1]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    axes[0].set_title("Document embeddings (LSA, 2D view)", fontweight="bold")
    sim = cosine_similarity(Z)
    im = axes[1].imshow(sim, cmap="magma", vmin=-0.2, vmax=1)
    axes[1].set_title("Cosine similarity", fontweight="bold")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    doc_df, word_df, msg = pd.DataFrame(), pd.DataFrame(), f"Embedding size: **{k}** dimensions from {len(docs)} documents."
    if (query_text or "").strip():
        qz = svd.transform(vec.transform([query_text]))
        s = cosine_similarity(qz, Z)[0]
        top = np.argsort(-s)[:5]
        doc_df = pd.DataFrame([(f"D{i + 1}", round(float(s[i]), 3), docs[i][:100]) for i in top], columns=["Doc", "Similarity", "Text"])
    if (query_word or "").strip():
        terms = list(vec.get_feature_names_out())
        qw = query_word.strip().lower()
        if qw in terms:
            T = svd.components_.T
            s = cosine_similarity(T[terms.index(qw)].reshape(1, -1), T)[0]
            top = [i for i in np.argsort(-s) if terms[i] != qw][:8]
            word_df = pd.DataFrame([(terms[i], round(float(s[i]), 3)) for i in top], columns=["Neighbour word", "Similarity"])
        else:
            msg += f"  \nThe word '{qw}' is not in this corpus vocabulary."
    return fig, doc_df, word_df, msg


# %% Unit 4: sentiment, opinion mining, syntax and semantics
def _lexicon():
    sia = get_resource("vader")
    if sia is not None:
        return sia, sia.lexicon
    return None, FALLBACK_LEXICON


def _polarity(sia, lex, sentence):
    if sia is not None:
        return sia.polarity_scores(sentence)["compound"]
    vals = [lex.get(w, 0.0) for w in re.findall(r"[a-z']+", sentence.lower())]
    return float(np.tanh(sum(vals) / 4.0))


def sentiment_demo(text):
    text = (text or "").strip()
    if not text:
        return [("Enter some text.", None)], pd.DataFrame(), _empty_fig(), ""
    sia, lex = _lexicon()
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    rows = []
    for s in sents:
        c = _polarity(sia, lex, s)
        rows.append((s[:100], round(c, 3), "Positive" if c >= 0.05 else ("Negative" if c <= -0.05 else "Neutral")))
    df = pd.DataFrame(rows, columns=["Sentence", "Compound score", "Opinion"])
    segs = []
    for tok in re.findall(r"\w+|\W+", text):
        v = lex.get(tok.lower(), 0.0)
        segs.append((tok, "positive" if v >= 1.0 else ("negative" if v <= -1.0 else None)))
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    vals = [r[1] for r in rows]
    ax.bar(range(1, len(vals) + 1), vals, color=["#2ecc71" if v >= 0.05 else "#e74c3c" if v <= -0.05 else "#95a5a6" for v in vals])
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Sentence")
    ax.set_ylabel("Compound polarity")
    ax.set_title("Sentence-level opinion", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    overall = float(np.mean(vals))
    engine = "VADER lexicon" if sia is not None else "built-in fallback lexicon"
    summary = f"**Overall polarity {overall:+.2f}** ({'positive' if overall >= 0.05 else 'negative' if overall <= -0.05 else 'neutral'}) using the {engine}."
    return segs, df, fig, summary


def wordnet_demo(word, word2, sentence):
    wn = get_resource("wordnet")
    if wn is None:
        return pd.DataFrame(), "WordNet could not be loaded (no internet for the one-time download).", ""
    word = (word or "").strip().lower()
    rows = []
    for s in wn.synsets(word)[:8]:
        hyper = ", ".join(h.lemmas()[0].name() for h in s.hypernyms()[:2])
        rows.append((s.name(), s.pos(), s.definition(), ", ".join(l.name() for l in s.lemmas()[:5]), hyper))
    df = pd.DataFrame(rows, columns=["Synset", "POS", "Definition", "Synonyms (lexemes)", "Hypernyms"])
    sim_msg = ""
    w2 = (word2 or "").strip().lower()
    if word and w2:
        a, b = wn.synsets(word), wn.synsets(w2)
        if a and b:
            sim = a[0].wup_similarity(b[0])
            sim_msg = f"Wu-Palmer similarity between **{word}** and **{w2}**: **{(sim or 0):.2f}**  \n"
        else:
            sim_msg = "One of the words was not found in WordNet.  \n"
    wsd_msg = ""
    if (sentence or "").strip() and word:
        try:
            from nltk.wsd import lesk
            sense = lesk(re.findall(r"[a-z']+", sentence.lower()), word)
            wsd_msg = f"Lesk word-sense disambiguation for **{word}**: `{sense.name()}` - {sense.definition()}" if sense else f"No sense found for '{word}'."
        except Exception as e:
            wsd_msg = f"Word-sense disambiguation unavailable ({type(e).__name__})."
    return df, sim_msg + wsd_msg, ""


_SYNTAX = {"nlp": None, "tried": False}


def _syntax_model():
    if not _SYNTAX["tried"]:
        _SYNTAX["tried"] = True
        if HAS_SPACY:
            try:
                _SYNTAX["nlp"] = spacy.load("en_core_web_sm")
            except Exception:
                try:
                    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True, timeout=300, capture_output=True)
                    _SYNTAX["nlp"] = spacy.load("en_core_web_sm")
                except Exception:
                    _SYNTAX["nlp"] = None
    return _SYNTAX["nlp"]


def syntax_demo(text):
    nlp = _syntax_model()
    if nlp is None:
        return pd.DataFrame(), "<i>The English grammar model (en_core_web_sm) could not be loaded. It needs a one-time download.</i>"
    doc = nlp((text or "").strip() or "I took Lipitor and started having severe muscle pain.")
    df = pd.DataFrame(
        [(t.text, t.lemma_, t.pos_, t.dep_, t.head.text) for t in doc],
        columns=["Token", "Lemma", "POS", "Dependency", "Head"],
    )
    try:
        from spacy import displacy
        html = displacy.render(doc, style="dep", jupyter=False, options={"compact": True, "distance": 90})
        html = f"<div style='overflow-x:auto;background:white;border-radius:10px;padding:6px'>{html}</div>"
    except Exception:
        html = ""
    return df, html


# %% Unit 5: large language model (FLAN-T5) with safe lazy loading
_LLM = {}
LLM_TASKS = {
    "Extract side-effects": "List every drug side effect mentioned in this patient post, separated by commas.\n\nPost: {t}",
    "Summarise the post": "Summarise this patient post in one short sentence.\n\nPost: {t}",
    "Rate severity": "Is the side effect described in this patient post mild, moderate or severe? Answer with one word.\n\nPost: {t}",
    "Any adverse reaction?": "Does this patient report an adverse drug reaction? Answer yes or no.\n\nPost: {t}",
    "Custom prompt": "{t}",
}


def llm_run(text, task, custom_prompt, model_name):
    text = (text or "").strip()
    if not text:
        return "Enter a patient post first."
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        if model_name not in _LLM:
            tok = AutoTokenizer.from_pretrained(model_name)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            _LLM[model_name] = (tok, mdl.to(dev).eval(), dev)
        tok, mdl, dev = _LLM[model_name]
        if task == "Custom prompt":
            prompt = (custom_prompt or "").strip() or text
            prompt = prompt.replace("{post}", text)
        else:
            prompt = LLM_TASKS[task].format(t=text)
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(dev)
        with torch.no_grad():
            out = mdl.generate(**enc, max_new_tokens=64, num_beams=3)
        return tok.decode(out[0], skip_special_tokens=True) or "(empty answer)"
    except Exception as e:
        return f"The language model could not run here ({type(e).__name__}: {e}). In Colab, make sure internet is on and try again."


# %% Project results (from the CADEC experiment) and charts
RESULTS = pd.DataFrame(
    {
        "Entity": ["Drug", "ADR", "Finding", "Disease", "Symptom"],
        "Precision": [0.94, 0.65, 0.43, 0.33, 0.50],
        "Recall": [0.88, 0.65, 0.29, 0.12, 0.08],
        "F1": [0.91, 0.65, 0.35, 0.18, 0.14],
    }
)


def results_figures():
    fig1, ax = plt.subplots(figsize=(6.2, 3.8))
    colors = ["#3F51B5", "#FF5722", "#26A69A", "#AB47BC", "#FFA726"]
    bars = ax.bar(RESULTS["Entity"], RESULTS["F1"], color=colors)
    for b, v in zip(bars, RESULTS["F1"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1-score")
    ax.set_title("CADEC test set: F1 by entity type", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig1.tight_layout()

    fig2, ax = plt.subplots(figsize=(6.2, 3.8))
    labels, formal, informal = ["Overall", "Drug", "Disease"], [0.68, 0.92, 0.26], [0.65, 0.84, 0.0]
    x = np.arange(3)
    ax.bar(x - 0.18, formal, 0.36, label="Formal", color="#3F51B5")
    ax.bar(x + 0.18, informal, 0.36, label="Informal", color="#FF5722")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1-score")
    ax.legend()
    ax.set_title("Audit: formal vs informal text", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig2.tight_layout()
    return fig1, fig2


SYLLABUS_MAP = pd.DataFrame(
    [
        ("Unit 1", "Introduction, challenges in NLP, applications", "Home / Post Analyzer, About"),
        ("Unit 2", "Regex, edit distance, NER, n-gram models and smoothing", "Preprocessing Lab, Spelling & Edit Distance, N-gram Lab, Post Analyzer"),
        ("Unit 3", "Bag-of-Words, TF-IDF, one-hot, embeddings", "Features & Embeddings"),
        ("Unit 4", "Syntax, WordNet, WSD, sentiment and opinion mining", "Sentiment, Syntax & Semantics"),
        ("Unit 5", "LLMs, evaluation, ethics and bias", "LLM Assistant, Model Report"),
    ],
    columns=["Syllabus unit", "Topics covered", "Where in the app"],
)

# %% User interface
CSS = """
.ss-banner{background:linear-gradient(135deg,#0f2027 0%,#2c5364 55%,#1abc9c 100%);color:#fff;padding:26px 30px;border-radius:18px;margin-bottom:8px}
.ss-banner h1{margin:0;font-size:34px;letter-spacing:1px;color:#fff}
.ss-banner p{margin:6px 0 0;opacity:.9;font-size:15px;color:#fff}
.ss-chips span{display:inline-block;background:rgba(255,255,255,.16);border-radius:20px;padding:4px 12px;margin:10px 6px 0 0;font-size:12px;color:#fff}
"""

BANNER = """
<div class="ss-banner">
  <h1>SafeSignal</h1>
  <p>An NLP early-warning toolkit that finds hidden drug side-effects in patient-written posts.</p>
  <div class="ss-chips"><span>NER</span><span>Regex</span><span>Edit distance</span><span>N-grams</span><span>TF-IDF</span>
  <span>Embeddings</span><span>WordNet</span><span>Sentiment</span><span>LLM</span></div>
</div>
"""

EXAMPLES = [
    ["I took Lipitor and started having severe muscle pain and memory problems."],
    ["lipiter gave me bad headaches n dizzy all day, doc switched me to something else"],
    ["Crestor 10mg, no problems for months, then sudden chest pain and shortness of breath. Scary."],
    ["Prednisone for asthma caused weight gain, mood swings and insomnia."],
]


def _highlighted(label, colors):
    import gradio as gr
    for kwargs in ({"color_map": colors, "show_legend": True}, {"color_map": colors}, {}):
        try:
            return gr.HighlightedText(label=label, **kwargs)
        except TypeError:
            continue
    return gr.HighlightedText(label=label)


def build_app():
    import gradio as gr

    sentiment_colors = {"positive": "#2ecc71", "negative": "#e74c3c"}
    kwargs = {"title": "SafeSignal - NLP toolkit"}
    if "theme" in inspect.signature(gr.Blocks.__init__).parameters:
        kwargs["theme"] = gr.themes.Soft()
        kwargs["css"] = CSS

    with gr.Blocks(**kwargs) as demo:
        gr.HTML(BANNER)

        with gr.Tabs():
            # ---------------- 1. Post analyzer
            with gr.Tab("Post Analyzer"):
                gr.Markdown("Paste one patient post. SafeSignal highlights drugs and reactions and gives a safety-signal score.")
                with gr.Row():
                    with gr.Column(scale=3):
                        post_in = gr.Textbox(label="Patient post", lines=5, value=SAMPLE_POSTS[0])
                        engine_in = gr.Radio(ENGINES, value=ENGINES[0], label="Extraction engine")
                        go = gr.Button("Analyse post", variant="primary")
                        gr.Examples(examples=EXAMPLES, inputs=[post_in], label="Try an example")
                    with gr.Column(scale=2):
                        risk_out = gr.HTML()
                hl_out = _highlighted("Detected entities", LABEL_COLORS)
                tbl_out = gr.Dataframe(label="Entity table", wrap=True)
                gr.Markdown(engine_status())
                go.click(analyze_post, [post_in, engine_in], [hl_out, tbl_out, risk_out])

            # ---------------- 2. Batch signal detection
            with gr.Tab("Batch Signal Detection"):
                gr.Markdown("Upload a CSV (column named `text`) or paste one post per line to discover which drugs are linked to which reactions.")
                with gr.Row():
                    with gr.Column():
                        file_in = gr.File(label="CSV or TXT file", file_types=[".csv", ".txt"])
                        paste_in = gr.Textbox(label="...or paste posts (one per line)", lines=8, value="\n".join(SAMPLE_POSTS))
                        eng_b = gr.Radio(ENGINES, value=ENGINES[1], label="Extraction engine")
                        run_b = gr.Button("Run batch analysis", variant="primary")
                    with gr.Column():
                        sum_b = gr.Markdown()
                        plot_b = gr.Plot(label="Top drugs and reactions")
                pairs_b = gr.Dataframe(label="Drug - reaction pairs (potential signals)", wrap=True)
                detail_b = gr.Dataframe(label="Per-post results", wrap=True)
                dl_b = gr.File(label="Download results (CSV)")
                run_b.click(batch_analyze, [file_in, paste_in, eng_b], [sum_b, pairs_b, detail_b, plot_b, dl_b])

            # ---------------- 3. Preprocessing lab
            with gr.Tab("Preprocessing Lab"):
                gr.Markdown("**Unit 1-2:** tokenisation, stop-words, stemming, lemmatisation and regular expressions.")
                txt_p = gr.Textbox(label="Text", lines=3, value="I was taking 20 mg of Lipitor for 3 weeks and I'm having terrible muscle aches!!")
                with gr.Row():
                    c_low = gr.Checkbox(label="Lowercase", value=True)
                    c_pun = gr.Checkbox(label="Remove punctuation", value=True)
                    c_stop = gr.Checkbox(label="Remove stop-words", value=False)
                    c_lem = gr.Checkbox(label="Lemmatise", value=False)
                    c_stem = gr.Checkbox(label="Stem (Porter)", value=False)
                btn_p = gr.Button("Run pipeline", variant="primary")
                info_p = gr.Markdown()
                df_p = gr.Dataframe(label="Token-by-token view", wrap=True)
                btn_p.click(preprocess_text, [txt_p, c_low, c_pun, c_stop, c_stem, c_lem], [df_p, info_p])
                gr.Markdown("### Regular expression tester")
                with gr.Row():
                    preset = gr.Dropdown(list(REGEX_PRESETS), value="Dosage (e.g. 20 mg)", label="Preset")
                    custom = gr.Textbox(label="...or your own pattern", placeholder=r"\bmuscle\s+\w+")
                btn_r = gr.Button("Find matches")
                info_r = gr.Markdown()
                df_r = gr.Dataframe(label="Matches")
                btn_r.click(regex_tester, [txt_p, preset, custom], [df_r, info_r])

            # ---------------- 4. Spelling and edit distance
            with gr.Tab("Spelling & Edit Distance"):
                gr.Markdown("**Unit 2:** patients often misspell drug names. Edit distance lets SafeSignal still recognise them.")
                with gr.Row():
                    w_in = gr.Textbox(label="Possibly misspelt drug name", value="lipiter")
                    b_btn = gr.Button("Find closest drugs", variant="primary")
                out_cd = gr.Dataframe(label="Closest drugs")
                b_btn.click(closest_drugs, [w_in], [out_cd])
                gr.Markdown("### Edit distance matrix (dynamic programming)")
                with gr.Row():
                    s1 = gr.Textbox(label="String A", value="lipiter")
                    s2 = gr.Textbox(label="String B", value="lipitor")
                m_btn = gr.Button("Compute matrix")
                dist_out = gr.Markdown()
                mat_out = gr.Dataframe(label="Levenshtein matrix")

                def _matrix(a, b):
                    a, b = (a or "")[:18], (b or "")[:18]
                    return f"Levenshtein distance = **{levenshtein(a, b)}**", levenshtein_matrix(a, b).reset_index().rename(columns={"index": ""})

                m_btn.click(_matrix, [s1, s2], [dist_out, mat_out])

            # ---------------- 5. N-gram lab
            with gr.Tab("N-gram Lab"):
                gr.Markdown("**Unit 2:** build a language model from patient posts, predict the next word and measure perplexity.")
                with gr.Row():
                    with gr.Column():
                        corp = gr.Textbox(label="Training corpus (leave empty for built-in posts)", lines=6, value="\n".join(SAMPLE_POSTS))
                        order = gr.Radio(["Unigram", "Bigram", "Trigram"], value="Bigram", label="Model order")
                        kk = gr.Slider(0.01, 1.0, value=0.1, step=0.01, label="Add-k smoothing (k)")
                        seed = gr.Textbox(label="Start of a sentence", value="i took")
                        test = gr.Textbox(label="Sentence to score (perplexity)", value="I took Lipitor and got muscle pain.")
                        ng_btn = gr.Button("Run language model", variant="primary")
                    with gr.Column():
                        ng_plot = gr.Plot()
                        ng_info = gr.Markdown()
                ng_df = gr.Dataframe(label="Top next-word predictions")
                ng_btn.click(ngram_demo, [corp, order, kk, seed, test], [ng_df, ng_plot, ng_info])

            # ---------------- 6. Features and embeddings
            with gr.Tab("Features & Embeddings"):
                gr.Markdown("**Unit 3:** one document per line. Compare Bag-of-Words, TF-IDF and One-Hot, then explore embeddings.")
                feat_text = gr.Textbox(label="Documents", lines=6, value="\n".join(SAMPLE_POSTS[:8]))
                with gr.Row():
                    feat_m = gr.Radio(["Bag-of-Words (CountVectorizer)", "TF-IDF", "One-Hot (binary)"], value="TF-IDF", label="Representation")
                    big = gr.Checkbox(label="Add bigrams", value=False)
                    stp = gr.Checkbox(label="Remove stop-words", value=True)
                f_btn = gr.Button("Build feature matrix", variant="primary")
                f_info = gr.Markdown()
                f_plot = gr.Plot()
                f_df = gr.Dataframe(label="Feature matrix (top terms)")
                f_btn.click(feature_demo, [feat_text, feat_m, big, stp], [f_df, f_plot, f_info])
                gr.Markdown("### Embeddings (LSA) and similarity search")
                with gr.Row():
                    q_text = gr.Textbox(label="Find posts similar to...", value="muscle pain after statin")
                    q_word = gr.Textbox(label="Nearest words to...", value="pain")
                e_btn = gr.Button("Build embeddings")
                e_info = gr.Markdown()
                e_plot = gr.Plot()
                with gr.Row():
                    e_docs = gr.Dataframe(label="Most similar documents")
                    e_words = gr.Dataframe(label="Nearest words")
                e_btn.click(embedding_demo, [feat_text, q_text, q_word], [e_plot, e_docs, e_words, e_info])

            # ---------------- 7. Sentiment
            with gr.Tab("Sentiment & Opinion"):
                gr.Markdown("**Unit 4:** lexicon-based sentiment analysis and opinion mining on patient experiences.")
                sent_in = gr.Textbox(label="Review", lines=4, value="The Zocor lowered my cholesterol, which is great. But the muscle pain was terrible and I felt awful. Doctor says it should get better.")
                s_btn = gr.Button("Analyse sentiment", variant="primary")
                s_sum = gr.Markdown()
                s_hl = _highlighted("Opinion words", sentiment_colors)
                with gr.Row():
                    s_plot = gr.Plot()
                    s_df = gr.Dataframe(label="Sentence-level results", wrap=True)
                s_btn.click(sentiment_demo, [sent_in], [s_hl, s_df, s_plot, s_sum])

            # ---------------- 8. Syntax and semantics
            with gr.Tab("Syntax & Semantics"):
                gr.Markdown("**Unit 4:** part-of-speech tags, dependency parse, WordNet meanings and word-sense disambiguation.")
                syn_in = gr.Textbox(label="Sentence", value="I took Lipitor and started having severe muscle pain.")
                syn_btn = gr.Button("Parse sentence", variant="primary")
                syn_html = gr.HTML()
                syn_df = gr.Dataframe(label="Tokens, POS tags and dependencies")
                syn_btn.click(syntax_demo, [syn_in], [syn_df, syn_html])
                gr.Markdown("### WordNet")
                with gr.Row():
                    wn_w = gr.Textbox(label="Word", value="pain")
                    wn_w2 = gr.Textbox(label="Compare with", value="ache")
                    wn_s = gr.Textbox(label="Sentence for word-sense disambiguation", value="The pain in my knee gets worse at night")
                wn_btn = gr.Button("Look up")
                wn_msg = gr.Markdown()
                wn_df = gr.Dataframe(label="Synsets", wrap=True)
                wn_hidden = gr.Markdown(visible=False)
                wn_btn.click(wordnet_demo, [wn_w, wn_w2, wn_s], [wn_df, wn_msg, wn_hidden])

            # ---------------- 9. LLM
            with gr.Tab("LLM Assistant"):
                gr.Markdown(
                    "**Unit 5:** a small instruction-tuned LLM (FLAN-T5) reads the post and answers a question about it. "
                    "The first run downloads the model (about 1-2 minutes). Treat answers as suggestions: LLMs can make mistakes."
                )
                llm_in = gr.Textbox(label="Patient post", lines=4, value=SAMPLE_POSTS[1])
                with gr.Row():
                    llm_task = gr.Dropdown(list(LLM_TASKS), value="Extract side-effects", label="Task")
                    llm_model = gr.Dropdown(["google/flan-t5-base", "google/flan-t5-small", "google/flan-t5-large"], value="google/flan-t5-base", label="Model")
                llm_custom = gr.Textbox(label="Custom prompt (used only for 'Custom prompt'; write {post} to insert the text)", value="What medicine is mentioned in: {post}")
                llm_btn = gr.Button("Ask the model", variant="primary")
                llm_out = gr.Textbox(label="Model answer", lines=3)
                llm_btn.click(llm_run, [llm_in, llm_task, llm_custom, llm_model], [llm_out])

            # ---------------- 10. Model report
            with gr.Tab("Model Report"):
                gr.Markdown(
                    "### CADEC experiment results\n"
                    "spaCy NER (tok2vec + ner), trained from scratch on 1,000 CADEC posts and tested on 250. "
                    "These are the results reported in the project report; they come from the CADEC-trained model, not from the demo model used in this app."
                )
                fig_a, fig_b = results_figures()
                with gr.Row():
                    gr.Plot(value=fig_a)
                    gr.Plot(value=fig_b)
                gr.Dataframe(value=RESULTS, label="Precision / Recall / F1 by entity")
                gr.Markdown(
                    "**Key finding:** Drug and ADR are detected reliably, but rare classes suffer from class imbalance, "
                    "and Disease detection drops from 0.26 to 0.00 on informal, misspelt text.\n\n"
                    "**Ethics and limitations:** the tool supports human reviewers and is not a diagnosis; forum text is personal health "
                    "information; the data covers one forum, 12 drugs and English only; the model performs worse on casual writing, which can "
                    "disadvantage some patients."
                )

            # ---------------- 11. About
            with gr.Tab("About"):
                gr.Markdown(
                    "### SafeSignal - Detecting Hidden Drug Side-Effects from Patient Reviews using NLP\n"
                    "Course: Natural Language Processing (0304220301), Symbiosis Artificial Intelligence Institute  \n"
                    "Author: Sanobar Latif Shaikh (PRN 25030421039)  \n\n"
                    "### How this app maps to the syllabus"
                )
                gr.Dataframe(value=SYLLABUS_MAP, wrap=True)
    return demo


# %% Launch
def launch_app(demo=None, share=None):
    demo = demo or build_app()
    in_colab = "google.colab" in sys.modules
    share = in_colab if share is None else share
    try:
        sig = inspect.signature(demo.launch).parameters
    except Exception:
        sig = {}
    kwargs = {"share": share, "show_error": True}
    if "theme" in sig:  # newer Gradio versions take theme/css at launch time
        import gradio as gr
        kwargs["theme"] = gr.themes.Soft()
        kwargs["css"] = CSS
    demo.launch(**kwargs)


if __name__ == "__main__":
    launch_app()
