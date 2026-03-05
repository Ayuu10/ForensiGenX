# Narrative-Driven Scene Reconstruction

## Project Overview
This project builds an AI system that generates images from narrative text while strictly preserving spatial structures and object mappings. Traditional text-to-image models (like default Stable Diffusion pipelines) often ignore structural relationships, hallucinate objects, or misplace objects entirely. 

This pipeline resolves those flaws by parsing text functionally beforehand:
**Text → Structured Scene Graph → Spatial Layout → Controlled Image Generation**

By forcing an intermediate mapping stage (Scene Graph), the generative diffusion process aligns flawlessly with controllable layout grids resulting in accurate narrative renderings.

### Architecture Pipeline
The system operates as a 6-stage pipeline:
1. **Text Preprocessing:** Tokenization, NLP cleaning.
2. **Semantic Extraction:** Capturing objects, attributes, and precise relationships.
3. **Scene Graph Validation:** Rules-based validation ensuring scene accuracy.
4. **Spatial Layout Generation:** 2D bounding-box spatial engine coordinate math. 
5. **ControlNet Conditioning:** Visual representations injected as generative priors.
6. **Diffusion Generation:** The final rendered semantic image.

---

## Phase 1 Implementation: NLP Engineering Pipeline
*Assigned to: Person 1 (NLP Engineer)*

This repository currently implements **Stage 1 & 2** of the above pipeline, processing raw image text from the `Visual Genome` dataset into structured feature rules used for spatial parsing.

### What Has Been Built

1. **Dataset Download (`download_dataset.py`)**
   - Automatically downloads the full Visual Genome `region_descriptions.json.zip` dataset.
   - Outputs the extracted dataset to use in actual NLP processing.

2. **Core Extraction Engine (`process_visual_genome.py`)** 
   - **Text Cleanser**: Mapped regex sanitization cleaning out stray punctuation, invalid characters, and malformed spacing. Generates UUID references and writes the unified output to `cleaned_text.txt`.
   - **Part-of-Speech & Dependency Processing**: Utilizes the `spaCy` NLP library (`en_core_web_sm`) to tag linguistic hierarchies.
   - **Object Recognition**: Employs noun chunks for objects while actively filtering out abstract concepts (like "view", "time", "side") ensuring only concrete physical entities are mapped.
   - **Attribute Recognition**: Extracts qualifying adjectives specifically tied to root physical objects.
   - **Spatial Relationship Math**: Evaluates adjacent physical objects sequentially over the text. Uses strict word boundaries matched against prepositions (e.g., "next to", "in front of", "under") to map relationships tightly without greedy false-positives.
   - Generates the final compiled representation to `linguistic_features.json`.

### How to Run Locally

Install the required environment dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Initialize the actual dataset:
```bash
python download_dataset.py
```

Run the NLP Extractor:
```bash
python process_visual_genome.py
```

### Outputs Generated
- `dataset/region_descriptions.json`: The full Visual Genome region descriptions dataset.
- `cleaned_text.txt`: The isolated sentence phrases cleaned and mapped by their unique sentence ID.
- `linguistic_features.json`: The fully realized structured output array defining distinct objects, attributes, and inter-object relationships formatted for the layout engine.
