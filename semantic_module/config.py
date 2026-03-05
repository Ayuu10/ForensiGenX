"""
Configuration for the semantic module.

Narrative Driven Scene Reconstruction - canonical mappings and settings.
"""

# Canonical object normalization: synonym -> canonical form
OBJECT_MAP = {
    "man": "person",
    "woman": "person",
    "boy": "person",
    "girl": "person",
    "guy": "person",
    "individual": "person",
    "suspect": "person",

    "vehicle": "car",
    "automobile": "car",
    "auto": "car",

    "bike": "bicycle",
    "cycle": "bicycle",

    "pistol": "gun",
    "revolver": "gun",
}

# Abstract / useless objects to filter out (dataset artifacts)
ABSTRACT_OBJECTS = {
    "colour",
    "color",
    "edge",
    "shade",
    "area",
    "part",
    "side",
    "portion",
    "background",
    "scene",
    "image",
}

# Maximum objects per scene for layout engine
MAX_OBJECTS_PER_SCENE = 6

# Canonical spatial relation mapping: variant forms -> canonical form
RELATION_MAP = {
    "next_to": "near",
    "beside": "near",
    "close_to": "near",
    "near": "near",

    "left": "left_of",
    "left_of": "left_of",

    "right": "right_of",
    "right_of": "right_of",

    "above": "above",
    "over": "above",

    "below": "below",
    "under": "under",
    "underneath": "under",

    "on": "on",
    "on_top_of": "on",

    "inside": "inside",
    "in": "inside",
}

# Only spatial relations useful for layout generation
VALID_RELATIONS = {
    "near",
    "left_of",
    "right_of",
    "above",
    "below",
    "on",
    "under",
    "inside",
}
