# Naive Bayes Sentiment Classifier — Notes

## 1. Project goal

The goal is to implement a simple **Multinomial Naive Bayes sentiment classifier** from scratch.

The classifier learns from labeled sentences and predicts whether a new sentence is:

- `positive`
- `negative`
- `neutral/uncertain` when the positive and negative scores are effectively tied

This project is educational. The goal is to understand the full pipeline, not to build a production-grade sentiment model.

---

## 2. Bayes theorem

Bayes theorem:

$$
P(A|B)=\frac{P(B|A)P(A)}{P(B)}
$$

Meaning:

> \(P(A|B)\) is the probability of event \(A\) after observing that event \(B\) has occurred.

For sentiment analysis:

$$
P(positive|words)=\frac{P(words|positive)P(positive)}{P(words)}
$$

---

## 3. Prior, likelihood, and posterior

### Prior probability

The probability of a class before observing the sentence.

Example:

$$
P(positive)
$$

If 4 out of 8 training sentences are positive:

$$
P(positive)=\frac{4}{8}=0.5
$$

### Likelihood

The probability of observing a word or sentence given a class.

Example:

$$
P(word|positive)
$$

This is estimated from word counts in the training data.

### Posterior probability

The updated probability after observing evidence.

Example:

$$
P(positive|words)
$$

This answers:

> After seeing these words, how likely is the sentence to be positive?

---

## 4. Why we do not compute the exact posterior during classification

In classification, we compare:

$$
P(positive|words)
$$

against:

$$
P(negative|words)
$$

Using Bayes theorem:

$$
P(positive|words)=
\frac{P(words|positive)P(positive)}{P(words)}
$$

and:

$$
P(negative|words)=
\frac{P(words|negative)P(negative)}{P(words)}
$$

Both have the same denominator:

$$
P(words)
$$

So the denominator does not change which class has the larger score.

Therefore, we compare only:

$$
P(words|positive)P(positive)
$$

vs:

$$
P(words|negative)P(negative)
$$

This is called **proportional comparison**.

---

## 5. Data structures used

### Training data

The training data is a list of tuples:

```python
(sentence, label)
```

Example:

```python
("I love this movie", 1)
```

Label convention:

- `1` = positive
- `0` = negative

### Frequency dictionary

The frequency dictionary stores:

```python
(word, label) -> count
```

Example:

```python
("love", 1): 3
("bad", 0): 2
```

A dictionary is used because we need to map each `(word, label)` pair to a frequency value.

### Vocabulary

The vocabulary is a set of unique words seen during training.

A set is used because vocabulary cares about uniqueness, not frequency.

---

## 6. Important variable distinction

This distinction is very important:

```text
p_pos / p_neg = document-level prior probabilities
n_pos / n_neg = total word occurrences in each class
```

Example:

```python
p_pos = positive_documents / total_documents
```

But:

```python
n_pos = total number of word occurrences in positive documents
```

They are not the same thing.

---

## 7. Laplace smoothing

### Problem: zero probabilities

If a word never appeared in a class during training, then:

$$
P(word|class)=0
$$

When sentence probabilities are computed by multiplying word probabilities, one zero probability makes the whole sentence probability zero.

### Solution

Use Laplace smoothing:

$$
P(word|class)=
\frac{\text{word count}+1}
{\text{total words in class}+\text{vocabulary size}}
$$

The idea is to pretend that every word in the vocabulary has appeared once before.

- Adding `1` to the word count prevents zero probabilities.
- Adding vocabulary size `V` to the denominator compensates for the extra counts added across all vocabulary words.

---

## 8. Why we use logs

Naive Bayes multiplies word probabilities:

$$
P(sentence|class)=P(w_1|class)P(w_2|class)\dots P(w_n|class)
$$

But multiplying many small probabilities can become extremely small and cause numerical underflow.

So we use logs:

$$
\log(a \times b)=\log(a)+\log(b)
$$

The score becomes:

$$
\log(P(class))+\sum_i \log(P(w_i|class))
$$

The class with the larger log-score wins.

Important note:

Log-scores are usually negative. The better score is the larger one, meaning the one closer to zero.

Example:

```python
-5 > -10
```

So `-5` is the better score.

---

## 9. Final pipeline

```text
train_data
    ↓
preprocess
    ↓
build_freqs
    ↓
build_vocab
    ↓
compute_priors
    ↓
count_words_by_label
    ↓
train_naive_bayes
    ↓
score_sentence
    ↓
predict
    ↓
evaluate
```

---

## 10. Main functions

### `preprocess(text)`

Converts raw text into normalized tokens.

Steps:

- lowercase
- remove punctuation
- split into words

### `build_freqs(train_data)`

Builds:

```python
(word, label) -> count
```

### `build_vocab(freqs)`

Builds the set of unique words.

### `compute_priors(train_data)`

Computes:

```python
p_pos
p_neg
```

### `count_words_by_label(freqs)`

Computes:

```python
n_pos
n_neg
```

### `get_word_probability(...)`

Computes:

$$
P(word|class)
$$

using Laplace smoothing.

### `score_sentence(...)`

Computes the log-score of a sentence under one class.

It returns a score, not a final prediction.

### `predict(sentence, model)`

Computes positive and negative scores, compares them, and returns the prediction.

### `evaluate(test_data, model)`

Computes model accuracy on test data.

---

## 11. Limitations

This classifier is simple and educational.

Main limitations:

- The dataset is tiny.
- It ignores word order.
- It assumes words are conditionally independent given the class.
- Unknown words are ignored.
- It uses a simple bag-of-words representation.
- Accuracy alone is not enough for serious model evaluation.
