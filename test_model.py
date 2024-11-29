from main import load_model, segment_text
import time
import joblib
import matplotlib.pyplot as plt
from collections import Counter


SEPARATOR = "\u200b"


def cleanup_str(text: str):
    text = text.strip(SEPARATOR).strip()
    text = text.replace("  ", " ")  # clean up 2 spaces to 1
    text = text.replace(" ", "\u200b \u200b")  # ensure 200b around space
    # clean up
    text = text.replace("\u200b\u200b", "\u200b")  # clean up dupe 200b
    text = text.replace("\u200b\u200b", "\u200b")  # in case multiple

    # remove special characters
    text = text.replace("\u2028", "")  # line separator
    text = text.replace("\u200a", "")  # hair space
    text = text.strip().replace("\n", "").replace("  ", " ")
    return text

def load_model(filename='khmer_segmenter.joblib'):
    model_data = joblib.load(filename)
    return model_data['crf'], model_data['dictionary']

def test_segmentation():
    print("Loading model...")
    crf, dictionary = load_model()
    
    test_cases = [
        # Common phrases
        "ចង់ទៅផ្ទះ", "ផ្ទះខ្ញុំ", "ខ្ញុំដែល","ខ្ញុំស្រលាញ់ប្រទេសកម្ពុជា", "ប្រមុខការទូតកម្ពុជា ជួបពិភាក្សាទ្វេភាគីជាមួយ លោក វ៉ាង យី រដ្ឋមន្រ្តីការបរទេសចិន"]
    
    print("\n=== Testing Khmer Word Segmentation ===\n")
    
    for i, text in enumerate(test_cases, 1):
        cleaned_text = cleanup_str(text)  # Apply cleanup
        print(f"\nTest Case {i}:")
        print("Input:", text)
        print("Cleaned:", cleaned_text)
        start_time = time.time()
        result = segment_text(crf, cleaned_text, dictionary)
        elapsed = (time.time() - start_time) * 1000
        print(f"Processing time: {elapsed:.2f}ms")
        #print(result)
        print("-" * 50)

def analyze_feature_importance(crf):
    # Get feature weights
    feature_weights = [(k, v) for k, v in crf.state_features_.items()]
    sorted_features = sorted(feature_weights, key=lambda x: abs(x[1]), reverse=True)
    
    # Top 20 most important features
    top_features = sorted_features[:20]
    
    # Plot
    plt.figure(figsize=(15, 8))
    features, weights = zip(*top_features)
    plt.barh([str(f) for f in features], weights)
    plt.title('Top 20 Most Important Features')
    plt.xlabel('Weight')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    # Feature type analysis
    feature_types = Counter()
    for feature, _ in feature_weights:
        if 'char.code' in str(feature):
            feature_types['Character Code'] += 1
        elif 'bigram' in str(feature):
            feature_types['Bigram'] += 1
        elif 'trigram' in str(feature):
            feature_types['Trigram'] += 1
        elif 'dict_word' in str(feature):
            feature_types['Dictionary'] += 1
        elif 'is_consonant' in str(feature):
            feature_types['Consonant'] += 1
        elif 'is_vowel' in str(feature):
            feature_types['Vowel'] += 1
        elif 'is_diacritic' in str(feature):
            feature_types['Diacritic'] += 1
    
    return {
        'top_features': top_features,
        'feature_types': feature_types
    }

def analyze_transitions(crf):
    transitions = [(f"{k[0]}->{k[1]}", v) 
                  for k, v in crf.transition_features_.items()]
    sorted_transitions = sorted(transitions, key=lambda x: abs(x[1]), reverse=True)
    
    # Plot transitions
    plt.figure(figsize=(10, 6))
    trans_labels, trans_weights = zip(*sorted_transitions)
    plt.barh(trans_labels, trans_weights)
    plt.title('State Transitions')
    plt.tight_layout()
    plt.savefig('transitions.png')
    plt.close()
    
    return sorted_transitions

def print_analysis(feature_analysis, transitions):
    print("\nModel Analysis")
    print("=" * 50)
    
    print("\nTop 10 Most Important Features:")
    print("-" * 50)
    for feature, weight in feature_analysis['top_features'][:10]:
        print(f"{feature}: {weight:.4f}")
    
    print("\nFeature Type Distribution:")
    print("-" * 50)
    for ftype, count in feature_analysis['feature_types'].most_common():
        print(f"{ftype}: {count}")
    
    print("\nTop State Transitions:")
    print("-" * 50)
    for trans, weight in transitions[:5]:
        print(f"{trans}: {weight:.4f}")

if __name__ == "__main__":
    test_segmentation()
    # Load model
    crf, dictionary = load_model()
    
    # Analyze
    feature_analysis = analyze_feature_importance(crf)
    transitions = analyze_transitions(crf)
    
    # Print results
    print_analysis(feature_analysis, transitions) 
    