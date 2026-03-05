"""
Scene graph validation.

Narrative Driven Scene Reconstruction - structural and referential integrity checks.
"""

from config import VALID_RELATIONS


def validate_scene(scene_graph):
    """
    Validate a scene graph for structural correctness.

    Validation rules:
    1. Scene must contain at least one object
    2. Object names must be unique
    3. Relationships must reference valid object IDs
    4. subject_id != object_id
    5. Relation must exist in VALID_RELATIONS

    Args:
        scene_graph: Dictionary with "objects" and "relationships" keys.
            objects: list of {"id": int, "name": str, "attributes": list}
            relationships: list of {"subject_id": int, "relation": str, "object_id": int}

    Returns:
        True if all validation rules pass. Raises ValueError on failure.
    """
    objects = scene_graph.get("objects", [])
    relationships = scene_graph.get("relationships", [])

    # Rule 1: scene must contain at least one object
    if not objects:
        raise ValueError("Scene must contain at least one object")

    # Rule 2: object names must be unique
    names = [obj["name"] for obj in objects]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate object names found: {names}")

    valid_ids = {obj["id"] for obj in objects}

    for rel in relationships:
        subject_id = rel.get("subject_id")
        object_id = rel.get("object_id")
        relation = rel.get("relation", "")

        # Rule 3: relationships must reference valid object IDs
        if subject_id not in valid_ids:
            raise ValueError(
                f"subject_id {subject_id} not found in objects. Valid IDs: {sorted(valid_ids)}"
            )
        if object_id not in valid_ids:
            raise ValueError(
                f"object_id {object_id} not found in objects. Valid IDs: {sorted(valid_ids)}"
            )

        # Rule 4: subject_id != object_id
        if subject_id == object_id:
            raise ValueError(
                f"Relationship must not have subject_id == object_id ({subject_id})"
            )

        # Rule 5: relation must exist in VALID_RELATIONS
        if not relation or not str(relation).strip() or relation not in VALID_RELATIONS:
            raise ValueError(
                f"Relation '{relation}' not in VALID_RELATIONS: {sorted(VALID_RELATIONS)}"
            )

    return True
