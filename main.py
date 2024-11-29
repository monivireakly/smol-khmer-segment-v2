from sklearn_crfsuite import CRF
from pathlib import Path
import pandas as pd
import numpy as np
import glob
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn_crfsuite.metrics import flat_classification_report, flat_f1_score
from sklearn.metrics import confusion_matrix

def load_data(data_dir):
    all_texts = []
    
    # Load all txt files
    for file in glob.glob(f"{data_dir}/*.txt"):
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            # Handle both tab and comma separators
            if '\t' in text:
                words = [line.split('\t')[0] for line in text.split('\n') if line and not line.startswith('#')]
            else:
                words = [line.split(',')[0] for line in text.split('\n') if line and not line.startswith('#')]
            all_texts.extend(words)
    
    return all_texts

def word2features(text, i, dictionary):
    features = {
        'char': text[i],
        'char.code': ord(text[i]),
        'is_consonant': 0x1780 <= ord(text[i]) <= 0x17A2,
        'is_vowel': 0x17B6 <= ord(text[i]) <= 0x17C5,
        'is_diacritic': ord(text[i]) == 0x17D2,
        'is_subscript': 0x17CC <= ord(text[i]) <= 0x17D3,
        'has_bantoc': text[i] == '់',
        'has_nikahit': text[i] == 'ំ',
        'has_toandakhiat': text[i] == '៉',
        'is_coeng': text[i] == '្',
        'diacritic_sequence': get_diacritic_sequence(text, i),
        'position': i,  # Position in text
        'is_start': i == 0,
        'is_end': i == len(text) - 1,
    }
    
    # Wider context window
    for offset in [-3, -2, -1, 1, 2, 3]:
        if 0 <= i + offset < len(text):
            features[f'char[{offset}]'] = text[i + offset]
            features[f'char[{offset}].code'] = ord(text[i + offset])
            
    # N-gram features
    if i > 0:
        features['bigram_prev'] = text[i-1:i+1]
    if i < len(text) - 1:
        features['bigram_next'] = text[i:i+2]
    if i > 1:
        features['trigram_prev'] = text[i-2:i+1]
    if i < len(text) - 2:
        features['trigram_next'] = text[i:i+3]
        
    # Dictionary features with sliding window
    for length in range(1, 15):  # Increased window size
        for start in range(max(0, i-5), min(len(text), i+6)):
            if start + length <= len(text):
                substring = text[start:start+length]
                features[f'dict_word_{start-i}_{length}'] = substring in dictionary
    
    return features

def text2features(text, dictionary):
    return [word2features(text, i, dictionary) for i in range(len(text))]

def text2labels(text, word_boundaries):
    labels = ['O'] * len(text)
    for pos in word_boundaries:
        labels[pos] = 'B'  # Beginning of word
    return labels

def prepare_training_data(texts, dictionary):
    X = []  # Features
    y = []  # Labels
    
    # Create synthetic sentences by combining words
    for i in range(0, len(texts), 3):
        chunk = texts[i:i+3]
        combined = ''.join(chunk)
        
        # Mark boundaries
        labels = ['O'] * len(combined)
        pos = 0
        for word in chunk:
            labels[pos] = 'B'
            pos += len(word)
            
        X.append(text2features(combined, dictionary))
        y.append(labels)
    
    return X, y

def train_segmenter():
    texts = load_data("data")
    print(f"Loaded {len(texts)} training samples")
    
    # Create dictionary from the same training data
    dictionary = set(texts)  # Use training words as dictionary
    
    X, y = prepare_training_data(texts, dictionary)
    print(f"Created {len(X)} training samples")
    
    crf = CRF(
        algorithm='lbfgs',
        c1=0.15,
        c2=0.05,
        max_iterations=164,
        all_possible_transitions=True,
        min_freq=4,
        all_possible_states=True,
        period=10,
        epsilon=1e-6,
        verbose=True
    )
    
    print("\nTraining started...")
    crf.fit(X, y)
    
    log_training_metrics(crf, X, y)
    
    return crf, dictionary

def segment_text(crf, text, dictionary):
    features = [word2features(text, i, dictionary) for i in range(len(text))]
    labels = crf.predict([features])[0]
    
    # Debug info
    print("\nDebug Information:")
    print("Characters | Labels")
    print("-" * 30)
    for char, label in zip(text, labels):
        print(f"{char:^10} | {label:^6}")
    
    # Convert predictions to segmented text
    segmented = []
    current_word = []
    
    for i, (char, label) in enumerate(zip(text, labels)):
        if label == 'B' and i > 0:
            if current_word:
                segmented.append(''.join(current_word))
                print(f"Word break after: {''.join(current_word)}")
            current_word = [char]
        else:
            current_word.append(char)
            
    if current_word:
        segmented.append(''.join(current_word))
    
    result = ' | '.join(segmented)
    result = apply_post_rules(result)
    print(f"\nFinal segmentation after rules:\n{result}")
    return result

def apply_post_rules(segmented_text):
    rules = [
        # Rule 1: Don't split pronouns
        (r'(ខ្ញុំ|យើង|គាត់|នាង)', r'\1'),
        
        # Rule 2: Keep common compounds together
        (r'ប្រទេស\s*\|\s*កម្ពុជា', 'ប្រទេសកម្ពុជា'),
        (r'រាជ\s*\|\s*ធានី', 'រាជធានី'),
        
        # Rule 3: Don't split numbers and units
        (r'(\d+)\s*\|\s*(ដុល្លារ|រៀល|បាត)', r'\1\2'),
        
        # New rules based on transitions
        (r'([ក-អ]្[ក-អ])\s*\|\s*([ក-អ])', r'\1\2'),  # Keep coeng sequences together
        (r'([ក-អ])\s*\|\s*([្់ំ៉])', r'\1\2'),  # Keep diacritics with consonants
        (r'([ក-អ])\s*\|\s*([\u17B6-\u17C5])', r'\1\2'),  # Keep vowels with consonants using Unicode range
        
        # Compound word patterns
        (r'(រដ្ឋ|អគ្គ|មហា)\s*\|\s*([ក-អ]+)', r'\1\2'),
    ]
    
    result = segmented_text
    for pattern, replacement in rules:
        result = re.sub(pattern, replacement, result)
    return result

def save_model(crf, dictionary, filename='khmer_segmenter.joblib'):
    model_data = {
        'crf': crf,
        'dictionary': dictionary
    }
    joblib.dump(model_data, filename)
    print(f"Model saved to {filename}")

def load_model(filename='khmer_segmenter.joblib'):
    model_data = joblib.load(filename)
    return model_data['crf'], model_data['dictionary']

def log_training_metrics(crf, X, y):
    # Make predictions
    y_pred = crf.predict(X)
    
    # Calculate F1 score
    f1 = flat_f1_score(y, y_pred, average='weighted')
    
    # Basic counts
    total_samples = len(X)
    total_features = len(crf.state_features_)
    
    print("\nTraining Metrics:")
    print("-" * 50)
    print(f"Total training samples: {total_samples}")
    print(f"Total features: {total_features}")
    print(f"F1 Score: {f1:.4f}")

def get_diacritic_sequence(text, i, window=3):
    """Get sequence of diacritics in window"""
    start = max(0, i - window)
    end = min(len(text), i + window + 1)
    sequence = text[start:end]
    return ''.join(c for c in sequence if 0x17B6 <= ord(c) <= 0x17D3)

# Usage
if __name__ == "__main__":
    # Training
    crf, dictionary = train_segmenter()  # Get dictionary from training
    save_model(crf, dictionary)
    
    test_texts = [
        "ខ្ញុំឈ្មោះសុខា",
        "ខ្ញុំស្រលាញ់ប្រទេសកម្ពុជា", 
        "សូមជួយខ្ញុំផង",
        "តម្លៃ១០០ដុល្លារ",
        "ង៉ូវប៉ាវឡុង",
        "លីមុនីវីរះ"
    ]
    
    for text in test_texts:
        print("\nSegmenting:", text)
        segment_text(crf, text, dictionary)