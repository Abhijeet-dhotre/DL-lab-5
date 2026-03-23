"""
LSTM Next Word Prediction Model
================================
A deep learning model using LSTM to predict the next word in a sequence.
Uses TensorFlow/Keras for implementation.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import os


class NextWordPredictor:
    """
    LSTM-based next word prediction model.
    """

    def __init__(self, max_sequence_length=100, embedding_dim=64, lstm_units=128):
        """
        Initialize the next word predictor.

        Args:
            max_sequence_length: Maximum length of input sequences
            embedding_dim: Dimension of word embeddings
            lstm_units: Number of LSTM units in the layer
        """
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.tokenizer = Tokenizer()
        self.model = None
        self.vocab_size = 0

    def preprocess_text(self, text):
        """
        Preprocess the input text.

        Args:
            text: Input text string

        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = " ".join(text.split())
        return text

    def prepare_sequences(self, text):
        """
        Prepare training sequences from text.

        Args:
            text: Input text string

        Returns:
            X (input sequences), y (target words)
        """
        # Preprocess text
        text = self.preprocess_text(text)

        # Tokenize
        self.tokenizer.fit_on_texts([text])
        total_words = len(self.tokenizer.word_index) + 1
        self.vocab_size = total_words

        # Create input sequences
        input_sequences = []
        for line in text.split("\n"):
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[: i + 1]
                input_sequences.append(n_gram_sequence)

        # Pad sequences
        max_sequence_len = min(
            self.max_sequence_length, max([len(x) for x in input_sequences])
        )
        input_sequences = np.array(
            pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre")
        )

        # Create predictors and label
        X = input_sequences[:, :-1]
        y = input_sequences[:, -1]

        # One-hot encode the output
        y = to_categorical(y, num_classes=total_words)

        return X, y, max_sequence_len

    def build_model(self, max_sequence_len):
        """
        Build the LSTM model.

        Args:
            max_sequence_len: Length of input sequences

        Returns:
            Compiled Keras model
        """
        model = Sequential(
            [
                # Embedding layer: converts word indices to dense vectors
                Embedding(
                    input_dim=self.vocab_size,
                    output_dim=self.embedding_dim,
                    input_length=max_sequence_len - 1,
                ),
                # First LSTM layer with return sequences for stacking
                LSTM(
                    units=self.lstm_units,
                    return_sequences=True,
                    dropout=0.2,
                    recurrent_dropout=0.2,
                ),
                # Second LSTM layer
                LSTM(units=self.lstm_units, dropout=0.2, recurrent_dropout=0.2),
                # Dense hidden layer
                Dense(units=self.lstm_units, activation="relu"),
                Dropout(0.2),
                # Output layer with softmax activation
                Dense(units=self.vocab_size, activation="softmax"),
            ]
        )

        # Compile model
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        self.model = model
        return model

    def train(self, text, epochs=100, batch_size=64, validation_split=0.1):
        """
        Train the model on the given text.

        Args:
            text: Training text
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation

        Returns:
            Training history
        """
        # Prepare sequences
        X, y, max_sequence_len = self.prepare_sequences(text)

        # Build model
        self.build_model(max_sequence_len)

        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Total sequences: {len(X)}")
        print(f"Sequence length: {max_sequence_len - 1}")
        print(f"\nModel Summary:")
        self.model.summary()

        # Callbacks
        callbacks = [
            EarlyStopping(monitor="loss", patience=10, restore_best_weights=True),
            ModelCheckpoint("best_model.keras", monitor="loss", save_best_only=True),
        ]

        # Train
        history = self.model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
        )

        return history

    def predict_next_word(self, seed_text, num_words=1, temperature=1.0):
        """
        Predict next word(s) given a seed text.

        Args:
            seed_text: Initial text to start prediction
            num_words: Number of words to predict
            temperature: Sampling temperature (lower = more conservative)

        Returns:
            Generated text with predicted words
        """
        result = seed_text.lower()

        for _ in range(num_words):
            # Tokenize the current text
            token_list = self.tokenizer.texts_to_sequences([result])[0]

            # Pad sequence
            token_list = pad_sequences(
                [token_list], maxlen=self.model.input_shape[1], padding="pre"
            )

            # Get prediction probabilities
            predicted_probs = self.model.predict(token_list, verbose=0)[0]

            # Apply temperature
            if temperature != 1.0:
                predicted_probs = np.log(predicted_probs + 1e-8) / temperature
                predicted_probs = np.exp(predicted_probs)
                predicted_probs = predicted_probs / np.sum(predicted_probs)

            # Sample from the distribution
            predicted_index = np.random.choice(len(predicted_probs), p=predicted_probs)

            # Convert index to word
            output_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == predicted_index:
                    output_word = word
                    break

            if output_word:
                result += " " + output_word
            else:
                break

        return result

    def get_top_predictions(self, seed_text, top_k=5):
        """
        Get top-k predictions for the next word.

        Args:
            seed_text: Input text
            top_k: Number of top predictions to return

        Returns:
            List of (word, probability) tuples
        """
        # Tokenize
        token_list = self.tokenizer.texts_to_sequences([seed_text.lower()])[0]

        # Pad sequence
        token_list = pad_sequences(
            [token_list], maxlen=self.model.input_shape[1], padding="pre"
        )

        # Get predictions
        predicted_probs = self.model.predict(token_list, verbose=0)[0]

        # Get top-k indices
        top_indices = np.argsort(predicted_probs)[-top_k:][::-1]

        # Convert to words
        top_predictions = []
        for idx in top_indices:
            for word, index in self.tokenizer.word_index.items():
                if index == idx:
                    top_predictions.append((word, float(predicted_probs[idx])))
                    break

        return top_predictions

    def save_model(self, model_path="lstm_model.keras", tokenizer_path="tokenizer.pkl"):
        """
        Save the model and tokenizer.

        Args:
            model_path: Path to save the model
            tokenizer_path: Path to save the tokenizer
        """
        self.model.save(model_path)
        with open(tokenizer_path, "wb") as f:
            pickle.dump(self.tokenizer, f)
        print(f"Model saved to {model_path}")
        print(f"Tokenizer saved to {tokenizer_path}")

    def load_model(self, model_path="lstm_model.keras", tokenizer_path="tokenizer.pkl"):
        """
        Load a saved model and tokenizer.

        Args:
            model_path: Path to the saved model
            tokenizer_path: Path to the saved tokenizer
        """
        self.model = tf.keras.models.load_model(model_path)
        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        print(f"Model loaded from {model_path}")
        print(f"Tokenizer loaded from {tokenizer_path}")


def create_sample_data():
    """
    Create sample training data for testing.

    Returns:
        Sample text corpus
    """
    sample_text = """
    the quick brown fox jumps over the lazy dog
    the quick brown fox jumps over the lazy cat
    a quick brown fox jumps over a lazy dog
    the lazy dog sleeps all day
    the lazy cat sleeps all night
    a lazy dog is a happy dog
    the quick brown cat runs fast
    the quick brown dog runs fast
    a quick brown cat is a happy cat
    deep learning is a subset of machine learning
    machine learning is a subset of artificial intelligence
    artificial intelligence is the future of technology
    natural language processing is an important field
    neural networks are powerful tools for deep learning
    recurrent neural networks are used for sequential data
    long short term memory networks solve the vanishing gradient problem
    transformers have revolutionized natural language processing
    attention mechanisms allow models to focus on relevant parts
    python is a popular programming language for machine learning
    tensorflow and pytorch are popular deep learning frameworks
    """
    return sample_text.strip()


def main():
    """
    Main function to demonstrate the LSTM next word prediction model.
    """
    print("=" * 60)
    print("LSTM Next Word Prediction Model")
    print("=" * 60)

    # Create predictor
    predictor = NextWordPredictor(
        max_sequence_length=50, embedding_dim=64, lstm_units=128
    )

    # Get training data
    training_text = create_sample_data()

    print("\nTraining the model...")
    print("-" * 40)

    # Train model
    history = predictor.train(
        text=training_text, epochs=100, batch_size=16, validation_split=0.1
    )

    print("\n" + "=" * 60)
    print("Testing Next Word Predictions")
    print("=" * 60)

    # Test predictions
    test_phrases = [
        "the quick",
        "deep learning",
        "natural language",
        "neural networks are",
        "python is a",
    ]

    for phrase in test_phrases:
        print(f"\nSeed: '{phrase}'")

        # Get top predictions
        top_preds = predictor.get_top_predictions(phrase, top_k=5)
        print("Top 5 predictions:")
        for word, prob in top_preds:
            print(f"  - {word}: {prob:.4f}")

        # Generate completion
        completion = predictor.predict_next_word(phrase, num_words=5, temperature=0.7)
        print(f"Generated: '{completion}'")

    # Save model
    print("\n" + "=" * 60)
    print("Saving model...")
    predictor.save_model()

    print("\nModel training and testing complete!")


if __name__ == "__main__":
    main()
