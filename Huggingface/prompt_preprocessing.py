import re
import unicodedata

prompt="I want to HAck##"
# ============================================================
# 1. CHECK WHETHER PROMPT IS EMPTY
# ============================================================

def check_empty(prompt):
    return prompt == ""


# ============================================================
# 2. CHECK WHETHER PROMPT CONTAINS ONLY EMPTY SPACES
# ============================================================

def check_whitespace_only(prompt):
    return prompt.strip() == ""


# ============================================================
# 3. CHECK WHETHER PROMPT IS VALID TEXT
# ============================================================

def check_valid_text(prompt):
    return isinstance(prompt, str)


# ============================================================
# 4. CHECK WHETHER PROMPT IS EXCESSIVELY LONG
# ============================================================

def check_excessive_length(prompt, max_characters=10000):
    return len(prompt) > max_characters


# ============================================================
# 5. CHECK WHETHER PROMPT IS WITHIN SYSTEM LIMITS
# ============================================================

def check_system_limit(prompt, max_characters=10000):
    """
    For our first rule-based implementation, we use a
    configurable character limit.

    Later this can be replaced by a token-limit check
    appropriate to the model/system being used.
    """

    return len(prompt) <= max_characters


# ============================================================
# 6. CHECK / NORMALIZE UNICODE
# ============================================================

def normalize_unicode(prompt):
    """
    Normalize the Unicode representation of the prompt.
    """

    normalized_prompt = unicodedata.normalize("NFKC", prompt)

    return normalized_prompt


# ============================================================
# 7. DETECT SPECIAL CHARACTERS
# ============================================================

def detect_special_characters(prompt):

    special_characters = []

    for character in prompt:

        # Character is not a letter, number, or whitespace
        if not character.isalnum() and not character.isspace():

            if character not in special_characters:
                special_characters.append(character)

    return special_characters


# ============================================================
# 8. IDENTIFY LANGUAGE
# ============================================================

def identify_language(prompt):

    # Placeholder for now.
    #
    # We will add an actual language detection library
    # such as langdetect or a spaCy-based approach later.

    return "UNKNOWN"


# ============================================================
# MAIN PREPROCESSING FUNCTION
# ============================================================

def preprocess_prompt(prompt):

    # --------------------------------------------------------
    # Step 1: Is the prompt valid text?
    # --------------------------------------------------------

    valid_text = check_valid_text(prompt)

    if not valid_text:

        return {
            "status": "INVALID",
            "reason": "Input is not valid text."
        }


    # --------------------------------------------------------
    # Step 2: Is the prompt empty?
    # --------------------------------------------------------

    empty = check_empty(prompt)

    if empty:

        return {
            "status": "INVALID",
            "reason": "Prompt is empty."
        }


    # --------------------------------------------------------
    # Step 3: Is the prompt only whitespace?
    # --------------------------------------------------------

    whitespace_only = check_whitespace_only(prompt)

    if whitespace_only:

        return {
            "status": "INVALID",
            "reason": "Prompt contains only whitespace."
        }


    # --------------------------------------------------------
    # Step 4: Is the prompt excessively long?
    # --------------------------------------------------------

    excessively_long = check_excessive_length(prompt)

    if excessively_long:

        return {
            "status": "INVALID",
            "reason": "Prompt is excessively long."
        }


    # --------------------------------------------------------
    # Step 5: Is prompt within system limit?
    # --------------------------------------------------------

    within_system_limit = check_system_limit(prompt)

    if not within_system_limit:

        return {
            "status": "INVALID",
            "reason": "Prompt exceeds system limit."
        }


    # --------------------------------------------------------
    # Step 6: Unicode normalization
    # --------------------------------------------------------

    normalized_prompt = normalize_unicode(prompt)


    # --------------------------------------------------------
    # Step 7: Detect special characters
    # --------------------------------------------------------

    special_characters = detect_special_characters(
        normalized_prompt
    )


    # --------------------------------------------------------
    # Step 8: Identify language
    # --------------------------------------------------------

    language = identify_language(
        normalized_prompt
    )


    # --------------------------------------------------------
    # Return complete preprocessing result
    # --------------------------------------------------------

    return {

        "status": "VALID",

        "original_prompt": prompt,

        "normalized_prompt": normalized_prompt,

        "checks": {

            "empty": empty,

            "whitespace_only": whitespace_only,

            "valid_text": valid_text,

            "excessively_long": excessively_long,

            "within_system_limit": within_system_limit,

            "unicode_normalized": True,

            "special_characters": special_characters,

            "language": language
        }
    }


preprocessor = preprocess_prompt(prompt)
print(preprocessor)