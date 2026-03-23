# LSTM Next Word Prediction

A deep learning model using LSTM (Long Short-Term Memory) networks to predict the next word in a sequence.

## Features

- **LSTM Architecture**: Multi-layer LSTM with embedding and dropout layers
- **Text Preprocessing**: Automatic tokenization and sequence padding
- **Temperature Sampling**: Control prediction randomness (lower = more conservative)
- **Top-K Predictions**: Get the most likely next words with probabilities
- **Model Persistence**: Save and load trained models

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from lstm_next_word_prediction import NextWordPredictor

# Create predictor
predictor = NextWordPredictor()

# Train on your text
text = "your training text here..."
predictor.train(text, epochs=50)

# Predict next words
result = predictor.predict_next_word("the quick", num_words=3)
print(result)  # Output: "the quick brown fox"

# Get top predictions
top_words = predictor.get_top_predictions("deep learning", top_k=5)
print(top_words)
```

### Running the Demo

```bash
python lstm_next_word_prediction.py
```

## Model Architecture

```
Embedding Layer (64 dimensions)
    ↓
LSTM Layer (128 units, dropout=0.2)
    ↓
LSTM Layer (128 units, dropout=0.2)
    ↓
Dense Layer (128 units, ReLU)
    ↓
Dropout (0.2)
    ↓
Output Layer (vocab_size units, Softmax)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| max_sequence_length | 100 | Maximum length of input sequences |
| embedding_dim | 64 | Dimension of word embeddings |
| lstm_units | 128 | Number of LSTM units per layer |
| epochs | 100 | Number of training epochs |
| batch_size | 64 | Batch size for training |
| temperature | 1.0 | Sampling temperature (lower = more conservative) |

## API Reference

### NextWordPredictor

#### `train(text, epochs=100, batch_size=64, validation_split=0.1)`
Train the model on the given text corpus.

#### `predict_next_word(seed_text, num_words=1, temperature=1.0)`
Predict the next word(s) given a seed text.

#### `get_top_predictions(seed_text, top_k=5)`
Get the top-k most likely next words with their probabilities.

#### `save_model(model_path, tokenizer_path)`
Save the trained model and tokenizer to disk.

#### `load_model(model_path, tokenizer_path)`
Load a previously trained model and tokenizer.

## Example Output

```
Seed: 'deep learning'
Top 5 predictions:
  - is: 0.8521
  - has: 0.0432
  - can: 0.0321
  - uses: 0.0245
  - requires: 0.0198

Generated: 'deep learning is a subset of machine'
```

## License

MIT License
