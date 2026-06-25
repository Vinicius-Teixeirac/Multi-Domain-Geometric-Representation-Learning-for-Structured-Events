# src/config/schema/chosen_columns.py
"""
Columns retained from the full GDELT event schema.

Covers the six semantically relevant groups used by all model families:
actor identity (Actor1/2 Name + 8 attribute codes), actor geo (Lat/Long/FeatureID
for both actors and the action location), event date (Day), and label (QuadClass).
"""

__all__ = ["CHOSEN_COLUMNS"]

CHOSEN_COLUMNS  = [
    "GlobalEventID",
    
    "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
    "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",

    "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
    "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    
    "Actor1Geo_Lat", "Actor1Geo_Long", "Actor1Geo_FeatureID",
    "Actor2Geo_Lat", "Actor2Geo_Long","Actor2Geo_FeatureID",
    "ActionGeo_Lat", "ActionGeo_Long","ActionGeo_FeatureID",
     
    "Day",
    
    "QuadClass"
]

