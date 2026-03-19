# src/config/schema/columns_schema.py
"""
Here we define kind and missing treatment for CHOSEN_COLUMNS. Notice this is project-specific and changes in the chosen columns 
induce changes here.

- COLUMNS_SCHEMA

kind:
    - id          : identifier, never modeled
    - target      : prediction target
    - categorical : discrete symbol
    - numeric     : scalar numeric
    - geo         : geographic numeric (lat/long)
    - date        : temporal value

missing:
    - explicit    : replace NaN with explicit token / value
    - error   : add missingness error column
    - error       : raise if NaN present

"""

COLUMNS_SCHEMA = {
    # ------------------------------------------------------------------
    # Identifiers (never modeled)
    # ------------------------------------------------------------------
    "GlobalEventID": {
        "kind": "id",
        "missing": "error"
    },

    # ------------------------------------------------------------------
    # Temporal
    # ------------------------------------------------------------------
    "Day": {
        "kind": "date",
        "missing": "error",
    },

    # ------------------------------------------------------------------
    # Actors – categorical
    # ------------------------------------------------------------------
    "Actor1Name": {
        "kind": "categorical",
        "missing": "explicit",
    },
    "Actor1CountryCode": {
        "kind": "categorical",
        "missing": "explicit",
    },
    "Actor1KnownGroupCode": {
        "kind": "categorical",
        "missing": "explicit",
    },
    "Actor1EthnicCode": {
        "kind": "categorical",
        "missing": "explicit",
    },
    "Actor1Religion1Code": {
        "kind": "categorical",
        "missing": "explicit",
    },
    "Actor1Religion2Code": {
        "kind": "categorical",
        "missing": "explicit",
    },
    "Actor1Type1Code": {
        "kind": "categorical",
        "missing": "explicit",
    },
    "Actor1Type2Code": {
        "kind": "categorical",
        "missing": "explicit",
    },
    "Actor1Type3Code": {
        "kind": "categorical",
        "missing": "explicit",
    },

    "Actor2Name": {
        "kind": "categorical",
        "missing": "explicit",
    },
    "Actor2CountryCode": {
        "kind": "categorical",
        "missing": "explicit",
    },
    "Actor2KnownGroupCode": {
        "kind": "categorical",
        "missing": "explicit",
    },
    "Actor2EthnicCode": {
        "kind": "categorical",
        "missing": "explicit",
    },
    "Actor2Religion1Code": {
        "kind": "categorical",
        "missing": "explicit",
    },
    "Actor2Religion2Code": {
        "kind": "categorical",
        "missing": "explicit",
    },
    "Actor2Type1Code": {
        "kind": "categorical",
        "missing": "explicit",
    },
    "Actor2Type2Code": {
        "kind": "categorical",
        "missing": "explicit",
    },
    "Actor2Type3Code": {
        "kind": "categorical",
        "missing": "explicit",
    },

    # ------------------------------------------------------------------
    # Geography
    # ------------------------------------------------------------------
    "Actor1Geo_Lat": {
        "kind": "geo",
        "missing": "error",
    },
    "Actor1Geo_Long": {
        "kind": "geo",
        "missing": "error",
    },
    "Actor1Geo_FeatureID": {
        "kind": "categorical",
        "missing": "explicit",
    },
    "Actor2Geo_Lat": {
        "kind": "geo",
        "missing": "error",
    },
    "Actor2Geo_Long": {
        "kind": "geo",
        "missing": "error",
    },
    "Actor2Geo_FeatureID": {
        "kind": "categorical",
        "missing": "explicit",
    },
    "ActionGeo_Lat": {
        "kind": "geo",
        "missing": "error",
    },
    "ActionGeo_Long": {
        "kind": "geo",
        "missing": "error",
    },
    "ActionGeo_FeatureID": {
        "kind": "categorical",
        "missing": "explicit",
    },

    # ------------------------------------------------------------------
    # Target (never encoded)
    # ------------------------------------------------------------------
    "QuadClass": {
        "kind": "target",
        "missing": "error"
    },
}
