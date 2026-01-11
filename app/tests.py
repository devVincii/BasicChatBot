# imports
import spacy

#*************************
import numpy as np
import json
import os
import sys


class TextProcessor:
    def __init__(self, intents_path):
        """
        Initializes the processor with the spaCy medium model.
        The 'md' model is required because it contains word vectors (embeddings).
        """
        print("Loading spaCy model (en_core_web_md)...")
        try:
            # We disable 'ner' and 'parser' for speed if we only need vectors/lemmas
            # However, for medical apps, keeping NER might be useful later.
            self.nlp = spacy.load("en_core_web_md")
        except OSError:
            print("Error: Model 'en_core_web_md' not found.")
            print("Please run: python -m spacy download en_core_web_md")
            sys.exit(1)

        self.intents_path = intents_path
        self.intents = [] # Unique tags (labels)
        self.documents = [] # (text_vector, tag_index)
        
        # Training data placeholders
        self.X = None # Features (Vectors)
        self.y = None # Labels (Indices)

    def preprocess_text(self, text):
        """
        Replaces tokenization, lemmatization, and stemming.
        Returns a cleaned spaCy Doc object.
        """
        # 1. The 'nlp' call runs the entire pipeline (Tokenize -> Tag -> Lemma)
        doc = self.nlp(text.lower())
        
        # 2. Filter out stop words (is, the, a) and punctuation
        # We perform this check to ensure our vector isn't diluted by noise.
        # meaningful_tokens = [token for token in doc if not token.is_stop and not token.is_punct]
        
        # NOTE: For vectors, we usually just want the doc.vector. 
        # However, checking for empty inputs is good practice.
        return doc

    def get_vector(self, text):
        """
        Converts text into a 300-dimensional numerical vector.
        This replaces 'Bag of Words'.
        """
        doc = self.preprocess_text(text)
        
        # spacy's .vector attribute returns the average of all word vectors in the sentence.
        # This captures semantic meaning (Context).
        return doc.vector

    def load_data(self):
        """
        Loads intents and converts patterns directly to vectors.
        """
        if not os.path.exists(self.intents_path):
            raise FileNotFoundError(f"Intents file not found at: {self.intents_path}")

        with open(self.intents_path, 'r') as f:
            data = json.load(f)

        vectors = []
        labels = []

        print("Processing training data...")

        # 1. Collect all unique tags first to ensure consistent indexing
        for intent in data: # Assuming data is list of intents or dict with 'intents' key
            # Handle different JSON structures (list vs dict)
            iterator = data['intents'] if 'intents' in data else data
            
            for item in iterator:
                tag = item.get('tag')  # Using .get() is safer
                if not tag:
                    # In your medical_rules.json, keys are tags. We might need to adapt.
                    # This logic assumes standard intents.json format. 
                    # We will adapt to your specific medical_rules structure below.
                    continue
                
                if tag not in self.intents:
                    self.intents.append(tag)

        # 2. Process patterns
        # ADAPTATION: Your medical_rules.json is a dictionary, not a list of intents.
        # We need to handle your specific structure:
        if isinstance(data, dict) and "diet_plans" in data:
            print("Detected medical_rules.json structure.")
            
            # --- Processing Diet Plans ---
            for condition, details in data["diet_plans"].items():
                tag = condition # e.g., "gerd", "diabetes"
                if tag not in self.intents:
                    self.intents.append(tag)
                
                # Treat "trigger_keywords" as the training patterns
                for pattern in details["trigger_keywords"]:
                    vec = self.get_vector(pattern)
                    vectors.append(vec)
                    labels.append(self.intents.index(tag))
            
            # --- Processing Specialists (Optional mapping) ---
            # You can add logic here to train on specialist keywords too
            for symptom, _ in data["specialist_map"].items():
                # We might want a general 'specialist_lookup' tag or specific ones
                pass 

        elif isinstance(data, dict) and "intents" in data:
            # Standard chatbot intents.json structure
            for intent in data['intents']:
                tag = intent['tag']
                for pattern in intent['patterns']:
                    vec = self.get_vector(pattern)
                    vectors.append(vec)
                    labels.append(self.intents.index(tag))
        # update training data placeholders
        self.X = np.array(vectors)
        self.y = np.array(labels)
        
        print(f"Training data ready: {len(self.X)} patterns processed.")
        print(f"Features shape: {self.X.shape}") # Should be (n_samples, 300)

if __name__ == "__main__":
    # Example usage
    processor = TextProcessor("medical_rules.json")
    processor.load_data()
    
    # Test semantic similarity capability
    test_phrase = "sugar problem" # 'sugar' is in data, 'problem' is generic
    test_vec = processor.get_vector(test_phrase)
    
    print(f"\nVector representation of '{test_phrase}':")
    print(test_vec[:10]) # Print first 10 dimensions only
    
    