"""Project-wide constants shared across preprocessing, representation, and modeling."""

NUM_QUAD_CLASSES = 4  # CAMEO taxonomy: Verbal Cooperation, Material Cooperation, Verbal Conflict, Material Conflict

# Verbs cycled round-robin (by row position) in text_builder.event_to_text to
# phrase "[WHO] actor1 <verb> [WHOM] actor2" sentences for the BERT baseline.
INTERACTION_VERBS = [
    "interacted with",
    "engaged with",
    "was involved with"
]

# Sentinel string used in place of pandas NaN wherever a column's missing
# values must survive as an explicit category (see cleaning.MissingValueHandler
# "explicit" policy and text_builder's actor-name null checks).
NULL_TOKEN = "__NULL__"