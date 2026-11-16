"""Shared text-preprocessing and RegEx helpers for the Airline Tweet Sentiment project."""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

REQUIRED_NLTK_RESOURCES = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
]


def ensure_nltk_data() -> None:
    """Download required NLTK resources if they are not already present."""
    for resource_path, download_name in REQUIRED_NLTK_RESOURCES:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(download_name, quiet=True)


ensure_nltk_data()

_STOP_WORDS = set(stopwords.words("english")) - {"not", "no", "nor"}
_LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Clean a raw tweet for downstream feature extraction and classification.

    Steps:
    1. Lowercase the text.
    2. Remove @mentions and URLs; strip '#' from hashtags while keeping the word;
       remove everything that is not a letter or whitespace.
    3. Tokenise with nltk.word_tokenize().
    4. Remove English stop words while keeping 'not', 'no', and 'nor'.
    5. Lemmatize tokens with WordNetLemmatizer.
    6. Rejoin tokens into a single space-separated string.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    cleaned_tokens = [
        _LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in _STOP_WORDS and len(tok) > 1
    ]
    return " ".join(cleaned_tokens)


def extract_mentions_and_hashtags(text: str) -> tuple[list[str], list[str]]:
    """Return lists of @mentions and #hashtags found in a raw tweet."""
    if not isinstance(text, str):
        return [], []
    mentions = re.findall(r"@\w+", text)
    hashtags = re.findall(r"#\w+", text)
    return mentions, hashtags


def get_pos_tags(text: str) -> list[tuple[str, str]]:
    """Tokenize and POS-tag a tweet using NLTK's default tagger."""
    if not isinstance(text, str) or not text.strip():
        return []
    tokens = word_tokenize(text)
    return nltk.pos_tag(tokens)