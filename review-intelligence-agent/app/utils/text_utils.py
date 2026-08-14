"""
text_utils.py
-------------
Lightweight text cleaning and aspect/keyword extraction utilities.
Deliberately avoids heavy dependencies like spaCy so the app has a
small, reliable install footprint.
"""

import re
from collections import Counter

STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because been
before being below between both but by can't cannot could couldn't did didn't do does
doesn't doing don't down during each few for from further had hadn't has hasn't have
haven't having he he'd he'll he's her here here's hers herself him himself his how
how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most
mustn't my myself no nor not of off on once only or other ought our ours ourselves out
over own same shan't she she'd she'll she's should shouldn't so some such than that
that's the their theirs them themselves then there there's these they they'd they'll
they're they've this those through to too under until up very was wasn't we we'd we'll
we're we've were weren't what what's when when's where where's which while who who's
whom why why's with won't would wouldn't you you'd you'll you're you've your yours
yourself yourselves product item amazon review reviews really just also got get one
would could book books it's im ive dont didnt
""".split())

# Common product-review aspect vocabulary. Extend this list for your domain.
ASPECT_SEED_WORDS = {
    "price", "cost", "value", "quality", "delivery", "shipping", "packaging",
    "battery", "screen", "sound", "size", "fit", "material", "design",
    "customer service", "support", "warranty", "durability", "performance",
    "comfort", "taste", "smell", "color", "instructions", "setup", "software",
    "app", "connection", "battery life", "camera", "speed", "storage",
}


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str):
    return [w for w in text.split() if w not in STOPWORDS and len(w) > 2]


def extract_aspects(texts, top_n: int = 15):
    """
    Extract the most frequently mentioned aspect terms across a list of
    (already cleaned) review texts. Combines:
      1. Matches against a seed vocabulary of common review aspects
      2. General high-frequency noun-like unigrams as a fallback

    Returns a list of (aspect, count) tuples sorted by frequency.
    """
    if not texts:
        return []

    counter = Counter()
    for t in texts:
        tokens = set(_tokenize(t))
        for seed in ASPECT_SEED_WORDS:
            if " " in seed:
                if seed in t:
                    counter[seed] += 1
            elif seed in tokens:
                counter[seed] += 1

    # fallback: general frequent words if seed matches are sparse
    if sum(counter.values()) < max(5, len(texts) // 4):
        word_counter = Counter()
        for t in texts:
            word_counter.update(_tokenize(t))
        for word, cnt in word_counter.most_common(30):
            if word not in counter:
                counter[word] = cnt

    return counter.most_common(top_n)
