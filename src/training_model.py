from preprocessing import (
    num_encoder_tokens,
    num_decoder_tokens,
    encoder_input_data,
    decoder_input_data,
    decoder_target_data,
)

from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model

from preprocessing import (
    num_encoder_tokens,
    num_decoder_tokens,
    encoder_input_data,
    decoder_input_data,
    decoder_target_data,
)

from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model

import os

# Model hyperparameters
dimensionality = 256
batch_size = 10
epochs = 100

# Encoder
encoder_inputs = Input(shape=(None, num_encoder_tokens))

encoder_lstm = LSTM(dimensionality, return_state=True)

encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)

encoder_states = [state_hidden, state_cell]

# Decoder
decoder_inputs = Input(shape=(None, num_decoder_tokens))

decoder_lstm = LSTM(dimensionality, return_sequences=True, return_state=True)

decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm(
    decoder_inputs, initial_state=encoder_states
)

decoder_dense = Dense(num_decoder_tokens, activation="softmax")

decoder_outputs = decoder_dense(decoder_outputs)

# Training model
training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile model
training_model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)

if __name__ == "__main__":

    print("\nModel Summary:\n")
    training_model.summary()

    print("\nStarting training...\n")

    history = training_model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        verbose=1,
    )

    os.makedirs("models/trained", exist_ok=True)

    training_model.save("models/trained/training_model.keras")

    print("\nTraining complete.")
    print("Model saved to:")
    print("models/trained/training_model.keras")
