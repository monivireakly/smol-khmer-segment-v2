# Model Analysis

## Top 10 Most Important Features
| Feature                                      | Weight   |
|----------------------------------------------|----------|
| ('trigram_prev:្ចខ', 'B')                      | 8.6367   |
| ('char:៕', 'B')                              | 6.0694   |
| ('diacritic_sequence:្ុុ', 'O')               | 5.9509   |
| ('char[-1]:្', 'O')                           | 5.8984   |
| ('diacritic_sequence:ុាើ្', 'B')             | 5.5290   |
| ('trigram_next:អាគ', 'B')                     | 5.5038   |
| ('is_vowel', 'O')                            | 5.1038   |
| ('dict_word_-5_12', 'B')                      | -4.8688  |
| ('dict_word_-5_12', 'O')                      | 4.8688   |
| ('char[1]:់', 'O')                            | 4.7003   |

---

## Feature Type Distribution
| Feature Type         | Count   |
|-----------------------|---------|
| Trigram              | 27,202  |
| Bigram               | 5,223   |
| Dictionary           | 297     |
| Character Code       | 2       |
| Consonant            | 2       |
| Diacritic            | 2       |
| Vowel                | 1       |

---

## Top State Transitions
| Transition  | Weight   |
|-------------|----------|
| B -> B      | -5.7100  |
| B -> O      | 0.1608   |
| O -> B      | 0.0337   |
| O -> O      | -0.0133  |
