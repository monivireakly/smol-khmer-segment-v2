# KHMER WORD SEGMENTATION TOOL

## Requirements
- Python 3.8+
- `sklearn-crfsuite`
---

## Quick Start
- **`test_model.py`**: Testing the model
To segment Khmer text:  
```python
def test_segmentation():
    print("Loading model...")
    crf, dictionary = load_model()
    
    test_cases = ["ខ្ញុំស្រលាញ់ប្រទេសកម្ពុជា"] #add more iff nedeed.
    
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

if __name__ == "__main__":
    test_segmentation()

```
## Training a New Model
Ensure `joblib` is installed.

---

## Key Files
- **`main.py`**: Main segmentation class
- **`test_model.py`**: Testing the model
- **`data/`**: Training data and dictionaries

---

## Model Parameters
- **c1**: 0.15 (L1 regularization)  
- **c2**: 0.05 (L2 regularization)  
- **max_iterations**: 164  
- **min_freq**: 4  
- **algorithm**: L-BFGS  

---

## Features Used
- Character n-grams
- Diacritic patterns
- Dictionary lookups
- Character type (consonant/vowel)
- Transition features

---

## Performance
- **F1 Score**: ~0.99  
- **Active Features**: ~41k  
- **Training Time**: ~3 mins on a standard CPU  

---

## Limitations
- Requires clean input text.
- May struggle with rare compound words.
- Numbers and foreign words need special handling.
---

---
I just want to break things down and see how things are constructed. khmernltk is a great tool that produces 
higher quality results but I want to see how much I can do with limited data. The algorithm works perfect except the data. With more refined words from other sources, the results will be better. 

Original dataset Excluding DATASET.txt: https://github.com/silnrsi/khmerlbdict/tree/master

Further:
I will try to do Byte Pair Encoding next.
