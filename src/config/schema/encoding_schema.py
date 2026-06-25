# src/config/schema/encoding_schema.py

"""
ENCODING_SCHEMA

Declarative specification of how columns are converted into numeric features.

Each column MUST define:
- type: semantic feature type
- method: encoding or transformation method
- params: method-specific configuration (optional)

This file contains NO logic and NO learned state.
"""

ENCODING_SCHEMA = {

    # ------------------------------------------------------------------
    # Temporal
    # ------------------------------------------------------------------
    "Day": {
        "type": "temporal",
        "method": "cyclical",
        "params": {
            "period": 365  # calendar days per year (GDELT Day column is YYYYMMDD)
        }
    },


    # ------------------------------------------------------------------
    # High-cardinality categorical (hashed)
    # ------------------------------------------------------------------
    "Actor1Name": {
        "type": "categorical",
        "method": "hash",
        "params": {
            "hash_dim": 65536  # 2^16 buckets; actor names are high-cardinality but not unique
        }
    },
    "Actor2Name": {
        "type": "categorical",
        "method": "hash",
        "params": {
            "hash_dim": 65536  # 2^16 buckets
        }
    },

    "Actor1Geo_FeatureID": {
        "type": "categorical",
        "method": "hash",
        "params": {
            "hash_dim": 1048576  # 2^20 buckets; GeoNames FeatureID is a very high-cardinality place code
        }
    },
    "Actor2Geo_FeatureID": {
        "type": "categorical",
        "method": "hash",
        "params": {
            "hash_dim": 1048576  # 2^20 buckets; GeoNames FeatureID is a very high-cardinality place code
        }
    },
    "ActionGeo_FeatureID": {
        "type": "categorical",
        "method": "hash",
        "params": {
            "hash_dim": 1048576  # 2^20 buckets; GeoNames FeatureID is a very high-cardinality place code
        }
    },

    # ------------------------------------------------------------------
    # medium-cardinality categorical (label)
    # ------------------------------------------------------------------
    "Actor1CountryCode": {
        "type": "categorical",
        "method": "label"
    },
    "Actor1KnownGroupCode": {
        "type": "categorical",
        "method": "label"
    },
    "Actor1EthnicCode": {
        "type": "categorical",
        "method": "label"
    },
    "Actor1Religion1Code": {
        "type": "categorical",
        "method": "label"
    },
    "Actor1Religion2Code": {
        "type": "categorical",
        "method": "label"
    },
    "Actor1Type1Code": {
        "type": "categorical",
        "method": "label"
    },
    "Actor1Type2Code": {
        "type": "categorical",
        "method": "label"
    },
    "Actor1Type3Code": {
        "type": "categorical",
        "method": "label"
    },

    "Actor2CountryCode": {
        "type": "categorical",
        "method": "label"
    },
    "Actor2KnownGroupCode": {
        "type": "categorical",
        "method": "label"
    },
    "Actor2EthnicCode": {
        "type": "categorical",
        "method": "label"
    },
    "Actor2Religion1Code": {
        "type": "categorical",
        "method": "label"
    },
    "Actor2Religion2Code": {
        "type": "categorical",
        "method": "label"
    },
    "Actor2Type1Code": {
        "type": "categorical",
        "method": "label"
    },
    "Actor2Type2Code": {
        "type": "categorical",
        "method": "label"
    },
    "Actor2Type3Code": {
        "type": "categorical",
        "method": "label"
    },

    # ------------------------------------------------------------------
    # Geography
    # ------------------------------------------------------------------

    "Actor1Geo_Lat": {
        "type": "geo",
        "method": "geodetic_cartesian",
        "params": {
            "scale": True
        }
    },
    "Actor1Geo_Long": {
        "type": "geo",
        "method": "geodetic_cartesian",
        "params": {
            "scale": True
        }
    },
    "Actor2Geo_Lat": {
        "type": "geo",
        "method": "geodetic_cartesian",
        "params": {
            "scale": True
        }
    },
    "Actor2Geo_Long": {
        "type": "geo",
        "method": "geodetic_cartesian",
        "params": {
            "scale": True
        }
    },
    "ActionGeo_Lat": {
        "type": "geo",
        "method": "geodetic_cartesian",
        "params": {
            "scale": True
        }
    },
    "ActionGeo_Long": {
        "type": "geo",
        "method": "geodetic_cartesian",
        "params": {
            "scale": True
        }
    },
}
