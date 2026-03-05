"""
Semantic parser - converts linguistic features into structured scene graphs.

Narrative Driven Scene Reconstruction - main pipeline.
"""

import json
import os

from config import MAX_OBJECTS_PER_SCENE
from normalizer import normalize_object, normalize_relation
from validator import validate_scene


def load_linguistic_features(filepath):
    """
    Load linguistic features from JSON file.

    Args:
        filepath: Path to linguistic_features.json

    Returns:
        List of feature dicts or single dict
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Support both list of scenes and single scene
    if isinstance(data, list):
        return data
    return [data]


def build_object_list(objects_raw, attributes_raw):
    """
    Build objects list with unique IDs and normalized names/attributes.

    Pipeline:
    - Normalize objects
    - Remove None (abstract) objects
    - Remove duplicates, limit to MAX_OBJECTS_PER_SCENE
    - Assign IDs after filtering

    Args:
        objects_raw: List of raw object names
        attributes_raw: Dict mapping object name -> list of attributes

    Returns:
        Tuple of (objects list, name_to_id mapping for normalized names)
    """
    # Normalize objects, remove None (abstract)
    objects = [normalize_object(name) for name in objects_raw]
    objects = [o for o in objects if o is not None]
    # Remove duplicates, limit to MAX_OBJECTS_PER_SCENE
    objects = list(dict.fromkeys(objects))[:MAX_OBJECTS_PER_SCENE]

    # Assign IDs after filtering, attach attributes
    result = []
    name_to_id = {}
    for i, norm_name in enumerate(objects, start=1):
        # Collect attributes from any raw name that mapped to this normalized name
        attrs = []
        for raw_name in objects_raw:
            if normalize_object(raw_name) == norm_name:
                attrs.extend(attributes_raw.get(raw_name, []) or [])
        attrs = [normalize_object(a) for a in attrs if normalize_object(a) is not None]
        attrs = list(dict.fromkeys(attrs))
        obj = {"id": i, "name": norm_name, "attributes": attrs}
        result.append(obj)
        name_to_id[norm_name] = i

    return result, name_to_id


def parse_relation(rel_str):
    """
    Parse a relation string "subject relation object" into components.

    Expects format: "subject relation object" (3 space-separated tokens).
    Relation may contain underscores (e.g. next_to, on_top_of).

    Args:
        rel_str: Raw relation string (e.g. "chair next_to desk")

    Returns:
        Tuple (subject_name, relation, object_name) or None if unparseable
    """
    tokens = rel_str.strip().split()
    if len(tokens) < 3:
        return None
    subject = tokens[0]
    relation = tokens[1]
    obj = tokens[2]
    return (subject, relation, obj)


def build_relationships(relations_raw, name_to_id):
    """
    Build relationships list with subject_id, object_id, normalized relation.

    Args:
        relations_raw: List of "subject relation object" strings
        name_to_id: Mapping from normalized object name to ID

    Returns:
        List of relationship dicts
    """
    relationships = []
    seen = set()

    for rel_str in relations_raw:
        parsed = parse_relation(rel_str)
        if parsed is None:
            continue
        subject_name, relation, object_name = parsed

        # Normalize names for lookup (handles "Chair" -> "chair")
        sub_norm = normalize_object(subject_name)
        obj_norm = normalize_object(object_name)

        # Skip if subject or object not in our objects
        if sub_norm not in name_to_id or obj_norm not in name_to_id:
            continue

        subject_id = name_to_id[sub_norm]
        object_id = name_to_id[obj_norm]

        # Skip self-relations
        if subject_id == object_id:
            continue

        relation_canonical = normalize_relation(relation)
        # Skip useless relations (e.g. "in", "of", "with") not in VALID_RELATIONS
        if relation_canonical is None:
            continue

        # Deduplicate identical relationships
        key = (subject_id, relation_canonical, object_id)
        if key in seen:
            continue
        seen.add(key)

        relationships.append(
            {"subject_id": subject_id, "relation": relation_canonical, "object_id": object_id}
        )

    return relationships


def process_single_scene(feature):
    """
    Convert one linguistic feature entry into a scene graph.

    Args:
        feature: Dict with id, objects, attributes, relations

    Returns:
        Scene graph dict
    """
    scene_id = feature.get("id", 0)
    objects_raw = feature.get("objects", [])
    attributes_raw = feature.get("attributes", {})
    relations_raw = feature.get("relations", [])

    # Step 2: Normalize object names, Step 3: Assign unique IDs, Step 4: Attach attributes
    objects, name_to_id = build_object_list(objects_raw, attributes_raw)

    # Step 5-7: Parse relations, normalize, convert names to IDs
    relationships = build_relationships(relations_raw, name_to_id)

    scene_graph = {
        "scene_id": scene_id,
        "objects": objects,
        "relationships": relationships,
    }

    return scene_graph


def run_pipeline():
    """
    Main pipeline: load linguistic features -> convert to scene graphs -> validate -> save.
    """
    # Resolve paths: script lives in semantic_module/, data in project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    input_path = os.path.join(project_root, "linguistic_features.json")
    output_path = os.path.join(project_root, "scene_graph.json")

    # Step 1: Load linguistic_features.json
    features = load_linguistic_features(input_path)

    scene_graphs = []
    skipped = 0
    for feature in features:
        # Steps 2-8: Build scene graph for this entry
        scene_graph = process_single_scene(feature)
        # Skip scenes with no valid objects (all abstract)
        if not scene_graph["objects"]:
            skipped += 1
            continue
        # Step 9: Validate scene graph
        try:
            validate_scene(scene_graph)
        except ValueError:
            skipped += 1
            continue
        scene_graphs.append(scene_graph)

    # Output: single graph if one input, else list
    output = scene_graphs[0] if len(scene_graphs) == 1 else scene_graphs

    # Step 10: Save scene_graph.json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(scene_graphs)} scene graph(s) to {output_path}" + (f" ({skipped} skipped)" if skipped else ""))


if __name__ == "__main__":
    run_pipeline()
