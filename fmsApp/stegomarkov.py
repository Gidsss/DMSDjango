import os
import markovify
import random
import numpy as np
from math import ceil, log2
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from numba import njit


# Efficient bitstream conversion using NumPy
def file_to_bitstream(file_path: str) -> str:
    """Convert file to binary bitstream using NumPy."""
    with open(file_path, 'rb') as file:
        file_data = np.frombuffer(file.read(), dtype=np.uint8)

    # Use NumPy's unpackbits for fast conversion
    bitstream = np.unpackbits(file_data)
    return ''.join(map(str, bitstream))


# Efficient function for converting bitstream back to file
def bitstream_to_file(bitstream: str, output_file: str):
    """Converts a binary bitstream back into a binary file."""
    byte_data = [bitstream[i:i+8] for i in range(0, len(bitstream), 8)]
    byte_array = bytearray(int(byte, 2) for byte in byte_data)
    with open(output_file, 'wb') as file:
        file.write(byte_array)

# Build Markov model
def build_model(json_file: str):
    """Takes in a JSON file path containing the Markov model and builds it."""
    with open(json_file, "r") as f:
        model_json = f.read()
    return markovify.Text.from_json(model_json)

def block_encode(bitstream: str, block_size: int) -> list:
    """ Encodes the bitstream into blocks of size `block_size`. Pads the last block if necessary. """
    # Calculate the length of padding needed for the last block (if it's not full)
    padding_length = block_size - (len(bitstream) % block_size) if len(bitstream) % block_size != 0 else 0
    padded_bitstream = bitstream + ('0' * padding_length)  # Pad with '0's

    # Split the bitstream into blocks of the specified size
    blocks = [padded_bitstream[i:i + block_size] for i in range(0, len(padded_bitstream), block_size)]
    return blocks

def block_decode(blocks: list) -> str:
    """ Decodes a list of blocks back into the bitstream. Removes padding if present. """
    # Concatenate all blocks back into a single bitstream
    bitstream = ''.join(blocks)

    # Remove any trailing zeros that were added for padding during encoding
    return bitstream.rstrip('0')

class Encoder:
    """
    Encodes a bitstream using a Markov Model.
    For basic usage, run `self.generate()` and get the generated output from `self.output_str`.
    """
    def __init__(self, model: markovify.Text, bitstream: str, block_size: int = 8, logging: bool = False):
        """
        Initializes a Markov Encoder.

        Parameters:
            model (markovify.Text): The Markov chain model to use in encoding
            bitstream (str): The bitstream to encode
            block_size (int): The block size for encoding bits at a time (default: 8)
            logging (bool): Optional parameter to enable logging
        """
        self.model = model
        self.block_size = block_size  # Set the block size for block encoding
        self.blocks = block_encode(bitstream, block_size)  # Encode the bitstream into blocks

        # Entry points from Markov chain's "___BEGIN__" keys
        self.entrypoints = [key[1] for key in model.chain.model.keys() if "___BEGIN__" in key][1:]
        self.current_gram = None
        self.output = []
        self.end_key = 0
        self.exhausted = True
        self.finished = False
        self.logging = logging

    @property
    def output_str(self):
        """ Returns the current state of the output string. """
        # Filter out any None values before joining
        return " ".join(token for token in self.output if token is not None)

    def step(self):
        """ Generates a new word for the output and appends it to the output string. """
        if self.finished:
            return

        char_limit = 20
        matrix_limit = 10

        # Choose new starting point
        if self.exhausted:
            self.exhausted = False
            next_token, removed, bit_length, encoded_index = self._consume_from_list(self.entrypoints)

            if next_token is None:
                self.finished = True
                return

            if self.logging:
                remaining = f"{''.join(self.blocks[:char_limit])}..." if len(self.blocks) > char_limit else ''.join(self.blocks)
                print(f"Entrypoint Chosen: {next_token}")
                print(f"\tMatrix Length: {len(self.entrypoints)}\tMax Bit Length: {bit_length}")
                print(f"\tBlocks: [{removed}]-{remaining}")
                print(f"\tEncoded Index: {encoded_index}\tToken: {next_token}")
                print()

            # Construct gram
            self.current_gram = ("___BEGIN__", next_token)

        # Get next word
        else:
            transitions = self._get_transitions(self.current_gram)
            if "___END__" in transitions:
                if self.logging:
                    remaining = f"{''.join(self.blocks[:char_limit])}..." if len(self.blocks) > char_limit else ''.join(self.blocks)
                    print(f"{self.current_gram}:")
                    print(f"\tPossible Transitions: {self._pretty_print_list(transitions, matrix_limit)}")
                    print(f"\tBlocks: []-{remaining}")
                    print("\t-- Exhausting --")
                    print()

                self.exhausted = True
                self.current_gram = None
                return

            next_token, removed, bit_length, encoded_index = self._consume_from_list(transitions)

            if next_token is None:
                self.finished = True
                return

            if self.logging:
                remaining = f"{''.join(self.blocks[:char_limit])}..." if len(self.blocks) > char_limit else ''.join(self.blocks)
                print(f"{self.current_gram}:")
                print(f"\tPossible Transitions: {self._pretty_print_list(transitions, matrix_limit)}")
                print(f"\tMatrix Length: {len(transitions)}\tMax Bit Length: {bit_length}")
                print(f"\tBlocks: [{removed}]-{remaining}")
                print(f"\tEncoded Index: {encoded_index}\tToken: {next_token}")
                print()

            # Construct gram
            next_gram = list(self.current_gram)
            next_gram.append(next_token)
            self.current_gram = tuple(next_gram[1:])

        # Add token to output if it's not None
        if next_token:
            self.output.append(next_token)

        # If no more blocks are left, inject the end key into the output
        if not self.blocks:
            self.end_key = len(removed)

            # Inject end key into output at a random position
            i = random.randint(0, len(self.output) - 1)
            char_key = chr(self.end_key + 97)  # Convert end key length to a character
            self.output[i] += char_key

            if self.logging:
                injected_word = self.output[i]
                print(f"Output: {self.output_str}")
                print(f"\tEnd Key: {self.end_key} ({char_key})")
                print(f"\tInjection Point: \"{injected_word[:-1]}\" at index {i}")
                print(f"\tInjection Preview: ... {' '.join(self.output[max(0, i - 2):i])} "
                      f"{f'|{injected_word}|'} "
                      f"{' '.join(self.output[i + 1: min(len(self.output) - 1, i + 3)])} ...")

            self.finished = True

    def generate(self):
        """ Consumes the entire bitstream and generates the output for it """
        while not self.finished:
            self.step()

    def _consume_from_list(self, lst):
        # Get max possible bit length based on the size of the list
        list_length = len(lst)
        bit_length = ceil(log2(list_length))
        if list_length < 2 ** bit_length:
            bit_length -= 1

        # Handle case where not enough bits are left to determine index
        if len(self.blocks) < bit_length:
            # Not enough bits to proceed, so mark as finished
            self.finished = True
            return None, None, None, None

        # Read bit stream to get index, but cap the index size to avoid large numbers
        block_segment = ''.join(self.blocks[:bit_length])  # Get the appropriate number of bits
        encoded_index = 0 if bit_length == 0 else min(int(block_segment, 2), list_length - 1)  # Cap index within bounds

        # Sanity check: Ensure the encoded_index is within the bounds of the list
        if encoded_index >= list_length:
            return None, None, None, None

        # Get next token based on the capped index
        next_token = lst[encoded_index]

        # Log the removed bit segment and update the blocks
        removed = ''.join(self.blocks[:bit_length])
        self.blocks = self.blocks[bit_length:]  # Update the blocks after consuming bits

        return next_token, removed, bit_length, encoded_index

    def _get_transitions(self, gram):
        trans_matrix = self.model.chain.model[gram]
        trans_matrix = sorted(trans_matrix.items(), key=lambda kv: (kv[1]), reverse=True)
        transitions = [i[0] for i in trans_matrix]
        return transitions

    @staticmethod
    def _pretty_print_list(lst, limit):
        lst = list(map(lambda s: f"'{s}'", lst))
        if not lst:
            return "None"
        elif len(lst) == 1:
            return lst[0]
        elif len(lst) == 2:
            return f"{lst[0]} and {lst[1]}"
        elif len(lst) <= limit:
            return ", ".join(lst[:-1]) + ", and " + lst[-1]
        else:
            truncated_list = lst[:limit]
            remaining_count = len(lst) - limit
            return ", ".join(truncated_list) + f", and {remaining_count} more"

class Decoder:
    """
    Decodes a steganographic text using a Markov Model.
    For basic usage, run `self.solve()` and get the generated output from `self.output`.
    """
    def __init__(self, model: markovify.Text, stega_text: str, block_size: int = 8, logging: bool = False):
        """
        Initializes a Markov Decoder. Run `self.solve()` and get the generated output from `self.output`.

        Parameters:
            model (markovify.Text): The Markov chain model to use in decoding
            stega_text (str): The steganographic text to decode
            block_size (int): Block size used during encoding (default: 8)
            logging (bool): Optional parameter to enable logging
        """
        self.model = model
        self.stega_text = stega_text.split(" ")
        self.block_size = block_size  # Block size used for decoding
        self.blocks = []  # List to collect decoded blocks
        self.entrypoints = [key[1] for key in model.chain.model.keys() if "___BEGIN__" in key][1:]
        self.endkey = 0
        self.current_gram = None
        self.index = 0
        self.exhausted = True
        self.finished = False
        self.logging = logging
        self.output = ""

    def step(self):
        """Consumes a word from the steganographic text and appends the appropriate bits to the output."""
        matrix_limit = 10
        char_limit = 20

        if self.index >= len(self.stega_text) - 1 and not self.exhausted:
            if self.logging:
                print(f"\nOutput:\t\t{self.output}")
            self.finished = True
            return

        if self.exhausted:
            token = self.stega_text[self.index]

            if token not in self.entrypoints:
                key_char = token[-1]
                token = token[:-1]
                self.endkey = ord(key_char) - 97

            self.current_gram = ("___BEGIN__", token)
            embedded_index = self.entrypoints.index(self.current_gram[1])
            bit_length = ceil(log2(len(self.entrypoints)))
            if len(self.entrypoints) < 2 ** bit_length:
                bit_length -= 1
            bit_length = self.endkey if self.index == len(self.stega_text) - 1 else bit_length
            bit_string = bin(embedded_index)[2:].zfill(bit_length)

            self.blocks.append(bit_string)  # Append the block instead of the full bitstream
            self.exhausted = False
        else:
            transitions = self._get_transitions(self.current_gram)
            at_end = self.index == len(self.stega_text) - 1

            list_length = len(transitions)
            bit_length = ceil(log2(list_length))
            if list_length < 2 ** bit_length:
                bit_length -= 1
            bit_length = 0 if list_length == 1 else bit_length

            if "___END__" in transitions:
                self.exhausted = True
                self.current_gram = None
                self.index += 1
                return
            else:
                next_token = "" if at_end else self.stega_text[self.index + 1]

            if next_token not in transitions and not at_end:
                key_char = next_token[-1]
                self.endkey = ord(key_char) - 97
                next_token = next_token[:-1]

            bit_length = self.endkey if self.index == len(self.stega_text) - 2 else bit_length

            if bit_length != 0:
                embedded_index = transitions.index(next_token)
                bit_string = bin(embedded_index)[2:].zfill(bit_length)
            else:
                bit_string = ""

            self.blocks.append(bit_string)  # Append block to blocks list

            next_gram = list(self.current_gram)
            next_gram.append(next_token)
            self.current_gram = tuple(next_gram[1:])
            self.index += 1

    def solve(self):
        """Consumes the entire steganographic text and generates an output bitstream."""
        while not self.finished:
            self.step()

        # After processing all steps, decode the blocks into bitstream
        self.output = block_decode(self.blocks)

    def _get_transitions(self, gram):
        trans_matrix = self.model.chain.model[gram]
        trans_matrix = sorted(trans_matrix.items(), key=lambda kv: (kv[1]), reverse=True)
        transitions = [i[0] for i in trans_matrix]
        return transitions

    @staticmethod
    def _pretty_print_list(lst, limit):
        lst = list(map(lambda s: f"'{s}'", lst))
        if not lst:
            return "None"
        elif len(lst) == 1:
            return lst[0]
        elif len(lst) == 2:
            return f"{lst[0]} and {lst[1]}"
        elif len(lst) <= limit:
            return ", ".join(lst[:-1]) + ", and " + lst[-1]
        else:
            truncated_list = lst[:limit]
            remaining_count = len(lst) - limit
            return ", ".join(truncated_list) + f", and {remaining_count} more"
