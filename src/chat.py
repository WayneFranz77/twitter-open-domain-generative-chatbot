import numpy as np
import re

from evaluate_model import (
    encoder_model,
    decoder_model,
    num_decoder_tokens,
    num_encoder_tokens,
    input_features_dict,
    target_features_dict,
    reverse_target_features_dict,
    max_decoder_seq_length,
    max_encoder_seq_length,
)


class ChatBot:

    negative_commands = ("no", "nope", "nah", "not now", "sorry")

    exit_commands = ("quit", "pause", "exit", "goodbye", "bye", "later", "stop")

    def start_chat(self):
        user_response = input(
            "Hi! I'm a weather chatbot. Would you like to chat?\n"
        ).lower()

        if user_response in self.negative_commands:
            print("Okay. Have a great day!")
            return

        print("Great! Type 'quit' whenever you'd like to exit.\n")

        while True:
            user_input = input("You: ")

            if self.make_exit(user_input):
                print("Bot: Goodbye!")
                break

            response = self.generate_response(user_input)

            print("Bot:", response)

    def make_exit(self, reply):
        for command in self.exit_commands:
            if command in reply.lower():
                return True
        return False

    def string_to_matrix(self, user_input):

        tokens = re.findall(r"[\w']+|[^\s\w]", user_input)

        user_input_matrix = np.zeros(
            (1, max_encoder_seq_length, num_encoder_tokens), dtype="float32"
        )

        for timestep, token in enumerate(tokens):

            if token in input_features_dict:
                user_input_matrix[0, timestep, input_features_dict[token]] = 1.0

        return user_input_matrix

    def generate_response(self, user_input):

        states_value = encoder_model.predict(
            self.string_to_matrix(user_input), verbose=0
        )

        target_seq = np.zeros((1, 1, num_decoder_tokens))

        target_seq[0, 0, target_features_dict["<START>"]] = 1.0

        decoded_sentence = ""

        stop_condition = False

        while not stop_condition:

            output_tokens, hidden_state, cell_state = decoder_model.predict(
                [target_seq] + states_value, verbose=0
            )

            sampled_token_index = np.argmax(output_tokens[0, -1, :])

            sampled_token = reverse_target_features_dict[sampled_token_index]

            if sampled_token not in ["<START>", "<END>"]:
                decoded_sentence += " " + sampled_token

            if (
                sampled_token == "<END>"
                or len(decoded_sentence) > max_decoder_seq_length
            ):
                stop_condition = True

            target_seq = np.zeros((1, 1, num_decoder_tokens))

            target_seq[0, 0, sampled_token_index] = 1.0

            states_value = [hidden_state, cell_state]

        return decoded_sentence.strip()


weather_bot = ChatBot()
weather_bot.start_chat()
