# Twitter Open-Domain Generative Chatbot

## Overview

This project implements a Sequence-to-Sequence (Seq2Seq) Generative Chatbot using TensorFlow/Keras and Twitter conversation data.

The chatbot is trained on pairs of Twitter messages and responses and learns to generate replies to user input using an Encoder-Decoder LSTM architecture.

This project was completed as part of the Codecademy "Generative Chatbots" curriculum and demonstrates foundational Natural Language Processing (NLP) and Deep Learning concepts such as:

- Text preprocessing
- Tokenization
- One-hot encoding
- Sequence-to-Sequence (Seq2Seq) models
- Encoder-Decoder architectures
- Teacher Forcing
- LSTM neural networks
- Model training and inference
- Interactive chatbot deployment

---

## Project Structure

```text
twitter-open-domain-generative-chatbot/
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   │   ├── weather.txt
│   │   ├── cat.txt
│   │   └── love.txt
│   ├── processed/
│   └── embeddings/
│
├── docs/
├── experiments/
├── logs/
│
├── models/
│   ├── checkpoints/
│   └── trained/
│       └── training_model.keras
│
├── notebooks/
│
├── outputs/
│   ├── evaluations/
│   └── predictions/
│
├── saved_tokenizers/
├── vectorizers/
│
├── src/
│   ├── __init__.py
│   ├── twitter_prep.py
│   ├── preprocessing.py
│   ├── training_model.py
│   ├── evaluate_model.py
│   └── chat.py
│
├── tests/
│   └── test_chatbot.py
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

---

## Dataset

The project includes three domain-specific Twitter datasets:

| Dataset | Description |
|----------|-------------|
| weather.txt | Weather-related conversations |
| cat.txt | Cat-related conversations |
| love.txt | Love and relationship conversations |

Each dataset contains conversational Twitter response pairs:

```text
Tweet A
Tweet B

Tweet C
Tweet D
```

where the second tweet represents the response to the first.

---

## Model Architecture

### Encoder

The encoder processes the user's input sequence and compresses it into hidden-state vectors.

```text
User Input
     ↓
Encoder LSTM
     ↓
Hidden State
Cell State
```

### Decoder

The decoder generates a response token-by-token using the encoder's final states.

```text
Encoder States
       ↓
Decoder LSTM
       ↓
Dense Layer
       ↓
Generated Response
```

### Hyperparameters

```python
latent_dim = 256
batch_size = 10
epochs = 100
```

---

## Technologies Used

### Languages

- Python 3

### Libraries

- TensorFlow
- Keras
- NumPy
- Regular Expressions (re)

### Concepts

- Deep Learning
- Natural Language Processing
- Seq2Seq Networks
- Teacher Forcing
- Recurrent Neural Networks
- LSTMs

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/twitter-open-domain-generative-chatbot.git

cd twitter-open-domain-generative-chatbot
```

Create and activate a virtual environment:

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

---

## Training the Model

Select a dataset in:

```python
twitter_prep.py
```

Example:

```python
data_path = "data/raw/weather.txt"
```

Run preprocessing:

```bash
python src/preprocessing.py
```

Train the model:

```bash
python src/training_model.py
```

The trained model will be saved to:

```text
models/trained/training_model.keras
```

---

## Running the Chatbot

Launch the chatbot:

```bash
python src/chat.py
```

Example:

```text
Hi! I'm a weather chatbot. Would you like to chat?

yes

Great! Type 'quit' whenever you'd like to exit.

You: How is the weather?

Bot: that that that that that
```

---

## Results

The chatbot successfully:

- Loads a trained Seq2Seq model
- Builds encoder inference models
- Builds decoder inference models
- Converts user input into one-hot encoded vectors
- Generates responses token-by-token
- Maintains an interactive conversation loop

### Limitations

Because the training datasets are intentionally small, the chatbot may:

- Repeat common words
- Produce nonsensical responses
- Overfit the training data
- Generate repetitive outputs

These limitations are expected given the educational nature of the project and the size of the datasets.

---

## Lessons Learned

Through this project I gained hands-on experience with:

- NLP preprocessing pipelines
- Sequence-to-Sequence architectures
- Encoder-Decoder inference models
- Keras model training and saving
- TensorFlow model loading
- One-hot encoded token representations
- Generative chatbot design
- Debugging deep learning applications

---

## Future Improvements

Potential enhancements include:

- Larger Twitter datasets
- Attention mechanisms
- Word embeddings (Word2Vec/GloVe)
- Beam search decoding
- Transformer architectures
- Intent classification
- Entity extraction
- Hybrid Retrieval + Generative responses
- Fine-tuning modern LLMs

---

## Project Status

✅ Complete

The chatbot successfully trains, saves, loads, and generates responses using a Seq2Seq LSTM model.

This project serves as a foundational NLP and Deep Learning implementation demonstrating the core principles of generative conversational AI.

---

## License

MIT License

Copyright (c) 2026 Wayne Franz

See the LICENSE file for details.