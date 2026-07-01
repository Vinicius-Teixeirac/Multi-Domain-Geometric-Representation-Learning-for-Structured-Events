"""Verbalizes structured GDELT event rows into natural-language sentences.

Turns each event's actor/geo/day columns into a tagged sentence
([WHO]/[WHOM]/[WHERE]/[WHEN]) for the BERT text-classification baseline;
build_event_texts is the entry point used by text_runner.py.
"""

import pandas as pd
from datetime import datetime
from typing import Optional

from src.utils.constants import INTERACTION_VERBS

__all__ = [
    "translate_code",
    "format_day",
    "normalize_name",
    "verbalize_actor",
    "verbalize_event_location",
    "event_to_text",
    "build_event_texts",
]

def translate_code(code: object, dictionary: dict) -> Optional[str]:
    """Return the human-readable label for a CAMEO code, or None if missing/null."""
    if code is None or pd.isna(code) or code == "__NULL__":
        return None
    return dictionary.get(code)


def format_day(day: object) -> Optional[str]:
    """Convert a GDELT YYYYMMDD or YYYYMM integer to a readable month-year string."""
    if day is None or pd.isna(day):
        return None

    try:
        day = str(int(day))
        if len(day) == 8:
            dt = datetime.strptime(day, "%Y%m%d")
        elif len(day) == 6:
            dt = datetime.strptime(day, "%Y%m")
        else:
            return None
        return dt.strftime("%B %Y")
    except Exception:
        return None


def normalize_name(name: str) -> str:
    """Title-case a name string for consistent display."""
    return str(name).title()


def verbalize_actor(row: object, prefix: str, dictionaries: dict) -> Optional[str]:
    """Build a natural-language actor phrase from GDELT actor columns.

    Layers name, role, descriptors (group/religion/ethnicity), home country,
    and current location in that precedence order.
    Returns None if the actor name is missing or null.
    """
    name = row.get(f"{prefix}Name")
    if pd.isna(name) or name == "__NULL__":
        return None

    name = normalize_name(name)
    name_lc = name.lower()

    role = translate_code(row.get(f"{prefix}Type1Code"),
                          dictionaries.get(f"{prefix}Type1Code", {}))

    known_group = translate_code(row.get(f"{prefix}KnownGroupCode"),
                                 dictionaries.get(f"{prefix}KnownGroupCode", {}))

    religion = translate_code(row.get(f"{prefix}Religion1Code"),
                              dictionaries.get(f"{prefix}Religion1Code", {}))

    ethnic = translate_code(row.get(f"{prefix}EthnicCode"),
                            dictionaries.get(f"{prefix}EthnicCode", {}))

    homeplace = translate_code(row.get(f"{prefix}CountryCode"),
                               dictionaries.get(f"{prefix}CountryCode", {}))

    geo = translate_code(row.get(f"{prefix}Geo_FeatureID"),
                         dictionaries.get(f"{prefix}Geo_FeatureID", {}))

    geo_country = translate_code(row.get(f"{prefix}Geo_CountryCode"),
                                 dictionaries.get(f"{prefix}Geo_CountryCode", {}))

    phrase = name

    if role:
        phrase = f"{phrase} ({role})"

    descriptors = []
    if known_group:
        descriptors.append(known_group)
    if religion:
        descriptors.append(religion.lower())
    if ethnic:
        descriptors.append(ethnic)

    if descriptors:
        phrase = f"{phrase}, a {' '.join(descriptors)} actor"

    if homeplace and homeplace.lower() not in name_lc:
        phrase = f"{phrase} from {homeplace}"

    if geo:
        geo = normalize_name(geo)
        if geo.lower() not in name_lc:
            phrase = f"{phrase}, while in {geo}"
    elif geo_country:
        if geo_country.lower() not in name_lc:
            phrase = f"{phrase}, while in {geo_country}"

    return phrase


def verbalize_event_location(row, dictionaries) -> Optional[str]:
    """Return a title-cased location name for the event's ActionGeo_FeatureID, or None."""
    loc = translate_code(row.get("ActionGeo_FeatureID"),
                         dictionaries.get("ActionGeo_FeatureID", {}))
    return normalize_name(loc) if loc else None


def event_to_text(row, dictionaries, verb: str) -> str:
    """
    Convert a single event row into a natural-language sentence.

    Constructs [WHO] actor1 verb [WHOM] actor2 [WHERE] location [WHEN] time
    phrases by calling verbalize_actor, verbalize_event_location, and format_day.

    Parameters
    ----------
    row : dict-like
        A single event row from the GDELT split DataFrame.
    dictionaries : dict
        Mapping from column name to {code: label} look-up dicts.
    verb : str
        Interaction verb (e.g. 'interacted with').

    Returns
    -------
    str
        Space-joined sentence with [WHO], [WHERE], [WHEN] tags.
    """
    actor1 = verbalize_actor(row, "Actor1", dictionaries)
    actor2 = verbalize_actor(row, "Actor2", dictionaries)

    sentences = []

    if actor1 and actor2:
        sentences.append(f"[WHO] {actor1} {verb} [WHOM] {actor2}.")
    elif actor1:
        sentences.append(f"[WHO] {actor1} was involved in an event.")
    elif actor2:
        sentences.append(f"[WHO] {actor2} was involved in an event.")

    location = verbalize_event_location(row, dictionaries)
    time = format_day(row.get("Day"))

    if location:
        sentences.append(f"[WHERE] {location}.")
    if time:
        sentences.append(f"[WHEN] {time}.")

    return " ".join(sentences)


def build_event_texts(df: pd.DataFrame, dictionaries: dict) -> list[str]:
    """
    Convert every row in df to a natural-language event sentence.

    Verbs are assigned deterministically (round-robin from INTERACTION_VERBS)
    so the same row always receives the same verb across pipeline runs.

    Parameters
    ----------
    df : pd.DataFrame
        GDELT split DataFrame (must contain actor and geo columns).
    dictionaries : dict
        Per-column code -> label look-up dicts.

    Returns
    -------
    list[str]
        One sentence per row, in the same order as df.
    """
    verbs = INTERACTION_VERBS
    texts = []

    for i, (_, row) in enumerate(df.iterrows()):
        verb = verbs[i % len(verbs)]  # deterministic
        texts.append(event_to_text(row, dictionaries, verb))

    return texts
