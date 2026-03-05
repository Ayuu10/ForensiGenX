import argparse
import json
import os
import re
import spacy
from tqdm import tqdm

# Load the English NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading English model for spaCy...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Abstract nouns to filter out
ABSTRACT_NOUNS = {
    "time", "day", "night", "morning", "evening", "part", "side", "area", "middle", 
    "center", "top", "bottom", "background", "foreground", "front", "back", "left", 
    "right", "way", "corner", "distance", "scene", "view", "reflection", "shadow"
}

# Relation patterns to look for
SPATIAL_RELATIONS = [
    "next to", "in front of", "on top of", "near", "on", "in", "under", "above", 
    "below", "beside", "behind", "over", "around", "inside", "outside", "between"
]

def clean_text(text):
    # Remove extra punctuation
    text = re.sub(r'[^\w\s\.-]', '', text)
    # Remove repeated spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def extract_features_from_doc(doc, original_clean_text):
    objects = set()
    attributes = {}
    
    # 1. Extract Objects using noun chunks
    for chunk in doc.noun_chunks:
        # Get the main noun of the chunk
        main_noun = chunk.root.text
        
        # Filter abstract nouns and pronouns
        if main_noun not in ABSTRACT_NOUNS and chunk.root.pos_ != "PRON":
            objects.add(main_noun)
            
            # 2. Extract Attributes
            # Find adjectives modifying the main noun
            adjs = [token.text for token in chunk.root.children if token.pos_ == "ADJ"]
            if adjs:
                attributes[main_noun] = adjs
                
    objects = list(objects)
    
    # 3. Extract Relations
    relations = []
    text_lower = original_clean_text
    
    # Sort objects by their position in text to easily find adjacent ones
    obj_spans = []
    for obj in objects:
        for match in re.finditer(rf"\b{re.escape(obj)}\b", text_lower):
            obj_spans.append({"obj": obj, "start": match.start(), "end": match.end()})
            
    obj_spans.sort(key=lambda x: x["start"])
    
    # Check adjacent pairs
    for i in range(len(obj_spans) - 1):
        span1 = obj_spans[i]
        span2 = obj_spans[i+1]
        
        between_text = text_lower[span1["end"]:span2["start"]].strip()
        for rel in SPATIAL_RELATIONS:
            if re.search(rf"\b{re.escape(rel)}\b", between_text):
                rel_formatted = rel.replace(" ", "_")
                relations.append(f"{span1['obj']} {rel_formatted} {span2['obj']}")
                break
                
    relations = list(set(relations))

    return {
        "objects": objects,
        "attributes": attributes,
        "relations": relations
    }

def main():
    parser = argparse.ArgumentParser(description="Process Visual Genome Dataset")
    parser.add_argument("--limit", type=int, default=1000, 
                        help="Limit the number of images to process (default: 1000). Set to 0 to process all.")
    args = parser.parse_args()

    dataset_path = "dataset/region_descriptions.json"
    
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found. Please run download_dataset.py first.")
        return

    print("Loading dataset into memory...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} images.")
    if args.limit > 0:
        print(f"Applying limit: Processing only the first {args.limit} images.")
        data = data[:args.limit]
    else:
        print("Processing all images (this might take a while).")

    print("Extracting subset and cleaning text...")
    
    sample_sentences = []
    cleaned_texts = []
    metadata = []
    
    sentence_id = 1
    
    # First pass: collect all phrases for batch SpaCy processing
    for image_data in tqdm(data, desc="Preparing Texts"):
        for region in image_data.get("regions", []):
            phrase = region.get("phrase", "")
            if not phrase:
                continue
                
            clean_phrase = clean_text(phrase)
            
            sample_sentence = {
                "id": sentence_id,
                "original_text": phrase,
                "clean_text": clean_phrase,
                "image_id": image_data.get("id"),
                "region_id": region.get("region_id")
            }
            sample_sentences.append(sample_sentence)
            cleaned_texts.append(f"{sentence_id}: {clean_phrase}")
            metadata.append(clean_phrase)
            
            sentence_id += 1

    print(f"Collected {len(metadata)} phrases. Running NLP Pipeline...")
    
    linguistic_features = []
    # Process texts in batches using nlp.pipe
    batch_size = 2000
    docs = nlp.pipe(metadata, batch_size=batch_size)
    
    for i, doc in enumerate(tqdm(docs, total=len(metadata), desc="NLP Processing")):
        clean_phrase = metadata[i]
        features = extract_features_from_doc(doc, clean_phrase)
        features["id"] = i + 1
        features["sentence"] = clean_phrase
        linguistic_features.append(features)

    # Save Outputs
    print("Saving outputs...")
    
    with open("sample_sentences.json", "w", encoding="utf-8") as f:
        json.dump(sample_sentences, f, indent=4)
        print("-> Saved sample_sentences.json")
        
    with open("cleaned_text.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned_texts))
        print("-> Saved cleaned_text.txt")
        
    with open("linguistic_features.json", "w", encoding="utf-8") as f:
        json.dump(linguistic_features, f, indent=4)
        print("-> Saved linguistic_features.json")
        
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
