"""
Normalization utilities for objects and relations.

Narrative Driven Scene Reconstruction - consistent naming and relation mapping.
"""

from config import OBJECT_MAP, ABSTRACT_OBJECTS, RELATION_MAP, VALID_RELATIONS


def normalize_object(name):
    """
    Normalize an object name for consistent representation.

    Steps:
    1. Lowercase
    2. Strip whitespace
    3. Map using OBJECT_MAP
    4. If name in ABSTRACT_OBJECTS return None
    5. Return normalized name

    Args:
        name: Raw object name string (e.g. "Wooden Chair ", "man")

    Returns:
        Normalized object name (e.g. "wooden chair", "person") or None if abstract
    """
    if not isinstance(name, str):
        name = str(name)
    key = name.lower().strip()
    # Map synonyms to canonical form
    canonical = OBJECT_MAP.get(key, key)
    # Remove abstract / useless objects
    if canonical in ABSTRACT_OBJECTS:
        return None
    return canonical


def normalize_relation(relation):
    """
    Normalize a relation string to its canonical form.

    Steps:
    1. Lowercase
    2. Map using RELATION_MAP
    3. If relation not in VALID_RELATIONS return None
    4. Return normalized relation

    Args:
        relation: Raw relation string (e.g. "next_to", "in")

    Returns:
        Normalized canonical relation (e.g. "near", "inside") or None if invalid
    """
    if not isinstance(relation, str):
        relation = str(relation)
    key = relation.lower().strip()
    canonical = RELATION_MAP.get(key, key)
    return canonical if canonical in VALID_RELATIONS else None
