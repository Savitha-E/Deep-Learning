import re
import unicodedata
from symspellpy import SymSpell


# ============================================================
# CONFIGURATION
# ============================================================

MAX_WORDS = 4000

DICTIONARY_PATH = (
    "/home/ibab/DL_data/dlvenv/lib/python3.10/"
    "site-packages/symspellpy/frequency_dictionary_en_82_765.txt"
)


# ============================================================
# INITIALIZE SPELL CHECKER
# ============================================================

sym_spell = SymSpell(
    max_dictionary_edit_distance=2,
    prefix_length=7
)

loaded = sym_spell.load_dictionary(
    DICTIONARY_PATH,
    term_index=0,
    count_index=1
)

if not loaded:
    raise FileNotFoundError(
        f"Could not load SymSpell dictionary:\n{DICTIONARY_PATH}"
    )


# ============================================================
# PROTECTED WORDS
# ============================================================
#
# These words should NEVER be automatically changed.
#
# You can add your own names, genes, drugs, abbreviations,
# institutions, etc. here.
# ============================================================

PROTECTED_WORDS = {
    "Savitha",
    "Atorvastatin",
    "Levothyroxine",
    "BRCA1",
    "BRCA2",
    "TP53",
    "RNA",
    "DNA",
    "RNA-seq",
    "CRISPR",
    "CRISPR-Cas9",
    "H3K4me3",
    "H3K27me3",
    "H3K36me3",
    "H3K27ac",
    "H3K9me3",
    "H3K4me1",
    "H3.3",
    "H3.1",
    "COVID-19",
}


# ============================================================
# COMMON CONTEXTUAL SPELLING CORRECTIONS
# ============================================================
#
# These are corrections that a normal dictionary cannot
# reliably determine from spelling alone.
#
# Example:
#
#     nam -> name
#     adn -> and
#     teh -> the
#
# IMPORTANT:
# Keep this dictionary conservative.
# ============================================================

COMMON_CORRECTIONS = {
    "teh": "the",
    "hte": "the",

    "adn": "and",
    "nad": "and",

    "taht": "that",
    "thta": "that",

    "isnt": "isn't",
    "dont": "don't",
    "doesnt": "doesn't",
    "cant": "can't",
    "wont": "won't",

    "whats": "what's",
    "whos": "who's",

    "recieve": "receive",
    "recieved": "received",
    "recieving": "receiving",

    "seperate": "separate",
    "definately": "definitely",
    "occured": "occurred",
    "occuring": "occurring",

    "becuase": "because",
    "beacuse": "because",

    "wich": "which",
    "whcih": "which",

    "thier": "their",
    "hteir": "their",

    "woudl": "would",
    "couldnt": "couldn't",
    "shouldnt": "shouldn't",

    "enviroment": "environment",
    "statment": "statement",

    "medecine": "medicine",
    "medcine": "medicine",

    "patinet": "patient",
    "paitent": "patient",

    "desease": "disease",
    "disese": "disease",

    "symtom": "symptom",
    "symptoms": "symptoms",

    "treatement": "treatment",
    "treament": "treatment",

    "dosagee": "dosage",

    "adversee": "adverse",
    "effectt": "effect",

    "commmon": "common",
    "comon": "common",

    "populaton": "population",
    "popultion": "population",

    "analaysis": "analysis",
    "anlaysis": "analysis",

    "experment": "experiment",
    "experiement": "experiment",

    "sequncing": "sequencing",
    "sequencng": "sequencing",

    "genom": "genome",
    "genmoe": "genome",

    "protiens": "proteins",
    "protien": "protein",

    "celss": "cells",
    "cel": "cell",

    "genee": "gene",
    "genes": "genes",

    # Specific example you tested
    "nam": "name",
}


# ============================================================
# WORD COUNT
# ============================================================

def count_words(text):
    """
    Count words in the prompt.
    """

    return len(re.findall(r"\S+", text))


# ============================================================
# CHECK PROTECTED WORD
# ============================================================

def is_protected(word):
    """
    Determine whether a word should not be corrected.
    """

    # Remove punctuation around word
    clean_word = word.strip(
        ".,!?;:\"'()[]{}"
    )

    if not clean_word:
        return True

    # Exact protected word
    if clean_word in PROTECTED_WORDS:
        return True

    # Case-insensitive protected word check
    for protected in PROTECTED_WORDS:

        if clean_word.lower() == protected.lower():
            return True

    # --------------------------------------------------------
    # Numbers
    # --------------------------------------------------------

    if any(char.isdigit() for char in clean_word):
        return True

    # --------------------------------------------------------
    # Scientific identifiers
    #
    # Examples:
    # H3K27me3
    # BRCA1
    # TP53
    # IL6
    # COVID-19
    # --------------------------------------------------------

    if re.search(
        r"[A-Za-z]+\d+|\d+[A-Za-z]+",
        clean_word
    ):
        return True

    # --------------------------------------------------------
    # Words containing uppercase letters
    #
    # This protects:
    # Savitha
    # BRCA
    # DNA
    # RNA
    # --------------------------------------------------------

    if any(char.isupper() for char in clean_word):
        return True

    # --------------------------------------------------------
    # Hyphenated technical terms
    # --------------------------------------------------------

    if "-" in clean_word:
        return True

    # --------------------------------------------------------
    # Very short words
    # --------------------------------------------------------

    if len(clean_word) <= 2:
        return True

    return False


# ============================================================
# PRESERVE CAPITALIZATION
# ============================================================

def preserve_capitalization(original, corrected):
    """
    Preserve the capitalization style of the original word.
    """

    if original.isupper():
        return corrected.upper()

    if original.istitle():
        return corrected.capitalize()

    return corrected


# ============================================================
# CORRECT SINGLE WORD
# ============================================================

def correct_word(word):
    """
    Correct one word conservatively.
    """

    # --------------------------------------------------------
    # Separate punctuation
    # --------------------------------------------------------

    match = re.match(
        r"^([^A-Za-z]*)([A-Za-z]+)([^A-Za-z]*)$",
        word
    )

    if not match:
        return word

    prefix, core, suffix = match.groups()

    # --------------------------------------------------------
    # Never modify protected words
    # --------------------------------------------------------

    if is_protected(core):
        return word

    lower_word = core.lower()

    # --------------------------------------------------------
    # Check manually defined corrections first
    # --------------------------------------------------------

    if lower_word in COMMON_CORRECTIONS:

        corrected = COMMON_CORRECTIONS[lower_word]

        corrected = preserve_capitalization(
            core,
            corrected
        )

        return prefix + corrected + suffix

    # --------------------------------------------------------
    # SymSpell correction
    # --------------------------------------------------------

    suggestions = sym_spell.lookup(
        lower_word,
        verbosity=0,
        max_edit_distance=2
    )

    if not suggestions:
        return word

    suggestion = suggestions[0]

    corrected = suggestion.term

    # --------------------------------------------------------
    # Don't change if identical
    # --------------------------------------------------------

    if corrected.lower() == lower_word:
        return word

    # --------------------------------------------------------
    # Avoid very uncertain corrections
    #
    # Only allow corrections where edit distance <= 2.
    # --------------------------------------------------------

    if suggestion.distance > 2:
        return word

    # --------------------------------------------------------
    # Preserve capitalization
    # --------------------------------------------------------

    corrected = preserve_capitalization(
        core,
        corrected
    )

    return prefix + corrected + suffix


# ============================================================
# CORRECT SPELLING
# ============================================================

def correct_spelling(text):
    """
    Correct spelling while protecting scientific terms,
    names, numbers and biomedical identifiers.
    """

    words = text.split()

    corrected_words = []

    for word in words:

        corrected_word_value = correct_word(word)

        corrected_words.append(
            corrected_word_value
        )

    return " ".join(corrected_words)


# ============================================================
# PREPROCESS PROMPT
# ============================================================

def preprocess_prompt(
    prompt,
    max_words=MAX_WORDS
):
    """
    Complete preprocessing pipeline.

    Steps:

    1. Check None
    2. Convert to string
    3. Strip whitespace
    4. Check empty
    5. Unicode normalization
    6. Normalize whitespace
    7. Count words
    8. Reject > 4000 words
    9. Correct spelling
    10. Unicode normalization again
    11. Return result
    """

    # ========================================================
    # STEP 1: NONE CHECK
    # ========================================================

    if prompt is None:

        return {
            "original_prompt": prompt,
            "processed_prompt": None,
            "status": "REJECTED",
            "reason": "EMPTY_PROMPT",
            "word_count": 0
        }

    # ========================================================
    # STEP 2: CONVERT TO STRING
    # ========================================================

    prompt = str(prompt)

    # ========================================================
    # STEP 3: STRIP WHITESPACE
    # ========================================================

    prompt = prompt.strip()

    # ========================================================
    # STEP 4: EMPTY CHECK
    # ========================================================

    if not prompt:

        return {
            "original_prompt": prompt,
            "processed_prompt": None,
            "status": "REJECTED",
            "reason": "EMPTY_PROMPT",
            "word_count": 0
        }

    # ========================================================
    # STEP 5: UNICODE NORMALIZATION
    # ========================================================

    prompt = unicodedata.normalize(
        "NFKC",
        prompt
    )

    # ========================================================
    # STEP 6: NORMALIZE WHITESPACE
    # ========================================================

    prompt = re.sub(
        r"\s+",
        " ",
        prompt
    ).strip()

    # ========================================================
    # STEP 7: COUNT WORDS
    # ========================================================

    word_count = count_words(prompt)

    # ========================================================
    # STEP 8: 4000 WORD LIMIT
    # ========================================================

    if word_count > max_words:

        return {
            "original_prompt": prompt,
            "processed_prompt": None,
            "status": "REJECTED",
            "reason": "TOO_LONG",
            "word_count": word_count
        }

    # ========================================================
    # STEP 9: SPELLING CORRECTION
    # ========================================================

    corrected_prompt = correct_spelling(prompt)

    # ========================================================
    # STEP 10: FINAL UNICODE NORMALIZATION
    # ========================================================

    corrected_prompt = unicodedata.normalize(
        "NFKC",
        corrected_prompt
    )

    # ========================================================
    # STEP 11: FINAL WHITESPACE CLEANUP
    # ========================================================

    corrected_prompt = re.sub(
        r"\s+",
        " ",
        corrected_prompt
    ).strip()

    # ========================================================
    # STEP 12: DETERMINE REASON
    # ========================================================

    if corrected_prompt != prompt:

        reason = "SPELLING_CORRECTED"

    else:

        reason = "OK"

    # ========================================================
    # STEP 13: RETURN RESULT
    # ========================================================

    return {
        "original_prompt": prompt,
        "processed_prompt": corrected_prompt,
        "status": "VALID",
        "reason": reason,
        "word_count": word_count
    }


# ============================================================
# FINAL FUNCTION
# ============================================================

def final_preprocess_prompt():

    prompt = input("Enter your prompt: ")

    result = preprocess_prompt(prompt)

    return result


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    result = final_preprocess_prompt()

    print(result)