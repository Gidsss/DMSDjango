import os
import random
from math import ceil, log2
import numpy as np
cimport numpy as np  # Cython import for NumPy
import markovify

# Function to build and save Markov models as JSON
def build_markov_json(str filename, str corpus_file, int state_size=2):
    """Takes in a corpus txt file and constructs a Markov model in json."""
    with open(corpus_file, "r", encoding="utf8") as f:
        # Construct the Markov model from the given corpus with the specified state size
        model = markovify.Text(f.read(), state_size=state_size)
    
    # Ensure the output directory exists
    os.makedirs("markov_models", exist_ok=True)
    
    # Save the model to a JSON file
    with open(f"markov_models/{filename}.json", "w") as f:
        f.write(model.to_json())
        
# Function to load the Markov model from a JSON file
def build_model(str json_file):
    """ Takes in a json markov model and builds it. Returns the constructed Markov Model. """
    with open(json_file, "r") as f:
        return markovify.Text.from_json(f.read())

# NumPy-based efficient bitstream conversion functions
def file_to_bitstream(str file_path) -> str:
    """Convert file to binary bitstream using NumPy."""
    with open(file_path, 'rb') as file:
        file_data = np.frombuffer(file.read(), dtype=np.uint8)

    # Use NumPy's unpackbits for fast conversion
    bitstream = np.unpackbits(file_data)
    return ''.join(map(str, bitstream))

def bitstream_to_file(str bitstream, str output_file):
    """Convert a binary bitstream back into a binary file."""
    byte_data = [bitstream[i:i+8] for i in range(0, len(bitstream), 8)]
    byte_array = bytearray(int(byte, 2) for byte in byte_data)
    with open(output_file, 'wb') as file:
        file.write(byte_array)

# Cython-optimized Encoder class
cdef class Encoder:
    """
    Encodes a bitstream using a Markov Model.
    For basic usage, run `self.generate()` and get the generated output from `self.output_str`.
    """
    cdef object model  # Markov model object
    cdef str bitstream  # Bitstream string
    cdef list entrypoints  # List of entry points for the Markov model
    cdef bint logging  # Logging flag
    cdef object current_gram  # Current n-gram in Markov model processing
    cdef list output_tokens  # Output tokens list
    cdef bint exhausted  # Flag to check if exhausted
    cdef bint finished  # Flag to check if finished
    cdef int end_key  # End key for the encoded message

    def __init__(self, object model, str bitstream, bint logging):
        self.model = model
        self.bitstream = bitstream
        self.logging = logging
        self.entrypoints = self._get_entrypoints()

        self.current_gram = None
        self.output_tokens = []
        self.exhausted = True
        self.finished = False
        self.end_key = 0

    def _get_entrypoints(self):
        """Get valid entry points from the Markov model."""
        if self.model.state_size == 1:
            return [key for key in self.model.chain.model.get(("___BEGIN__",)).keys()]
        else:
            return [key[-1] for key in self.model.chain.model.keys() if key.count("___BEGIN__") == self.model.state_size - 1][1:]

    @property
    def output(self):
        """Returns the current state of the output string."""
        return " ".join(self.output_tokens)

    def step(self):
        """Generates a new word for the output and appends it to the output string."""
        if self.finished:
            return

        if self.exhausted:
            self._choose_entrypoint()
        else:
            self._choose_next_token()

        if not self.bitstream:
            self._inject_end_key()

    def _choose_entrypoint(self):
        """Choose a new starting point (entrypoint) for the Markov chain."""
        self.exhausted = False
        next_token, removed, bit_length, encoded_index = self._consume_from_list(self.entrypoints)
        self.current_gram = (next_token,) if self.model.state_size == 1 else (*["___BEGIN__"] * (self.model.state_size - 1), next_token)

    def _choose_next_token(self):
        """Choose the next token in the Markov chain."""
        transitions = self._get_transitions(self.current_gram)
        if "___END__" in transitions:
            self.exhausted = True
            return
        next_token, removed, bit_length, encoded_index = self._consume_from_list(transitions)
        self.current_gram = (next_token,)

        if type(next_token) == tuple:
            self.output_tokens.extend(next_token)
        else:
            self.output_tokens.append(next_token)

    def _inject_end_key(self):
        """Inject the end key to mark the end of encoding."""
        self.end_key = len(self.bitstream)
        i = random.randint(0, len(self.output_tokens) - 1)
        self.output_tokens[i] += chr(self.end_key + 97)
        self.finished = True

    def generate(self):
        """Consumes the entire bitstream and generates the output for it."""
        while not self.finished:
            self.step()

    def _consume_from_list(self, lst):
        """Consume bits from the bitstream and choose an item from the list based on the bits."""
        list_length = len(lst)
        bit_length = ceil(log2(list_length))
        if list_length < 2 ** bit_length:
            bit_length -= 1

        encoded_index = 0 if bit_length == 0 else int(self.bitstream[:bit_length], 2)
        next_token = lst[encoded_index]
        removed = self.bitstream[:bit_length]
        self.bitstream = self.bitstream[bit_length:]

        return next_token, removed, bit_length, encoded_index

    def _get_transitions(self, gram):
        """Get possible transitions for the current gram in the Markov chain."""
        trans_matrix = self.model.chain.model[gram]
        trans_matrix = sorted(trans_matrix.items(), key=lambda kv: (kv[1]), reverse=True)
        transitions = [i[0] for i in trans_matrix]
        return transitions

    @staticmethod
    def _pretty_print_list(lst, limit):
        """Pretty print a list, showing only up to `limit` items."""
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


# Cython-optimized Decoder class
cdef class Decoder:
    """
    Decodes a steganographic text using a Markov Model.
    For basic usage, run `self.solve()` and get the generated output from `self.output`.
    """
    cdef object model  # Markov model object
    cdef str stega_text  # Steganographic text to decode
    cdef list entrypoints  # List of entry points for the Markov model
    cdef bint logging  # Logging flag
    cdef str output  # Decoded output

    cdef object current_gram  # Current gram in the Markov model
    cdef bint exhausted  # Flag to check if exhausted
    cdef bint finished  # Flag to check if finished
    cdef int index  # Index for processing the stega_text

    def __init__(self, object model, str stega_text, bint logging):
        self.model = model
        self.stega_text = stega_text.split(" ")
        self.logging = logging
        self.entrypoints = self._get_entrypoints()
        self.current_gram = None
        self.exhausted = True
        self.finished = False
        self.index = 0
        self.output = ""

    def _get_entrypoints(self):
        """Get valid entry points from the Markov model."""
        if self.model.state_size == 1:
            return [key for key in self.model.chain.model.get(("___BEGIN__",)).keys()]
        else:
            return [key[-1] for key in self.model.chain.model.keys() if key.count("___BEGIN__") == self.model.state_size - 1][1:]

    def step(self):
        """Consumes a word from the steganographic text and appends the appropriate bits to the output."""
        if self.finished:
            return

        if self.exhausted:
            self._choose_entrypoint()
        else:
            self._choose_next_token()

    def _choose_entrypoint(self):
        """Choose a new starting point (entrypoint) for the Markov chain."""
        self.exhausted = False
        token = self.stega_text[self.index]
        self.current_gram = (token,) if self.model.state_size == 1 else (*["___BEGIN__"] * (self.model.state_size - 1), token)

    def _choose_next_token(self):
        """Choose the next token in the Markov chain."""
        transitions = self._get_transitions(self.current_gram)
        next_token = self.stega_text[self.index + 1] if self.index < len(self.stega_text) - 1 else ""

        bit_length = ceil(log2(len(transitions)))
        if bit_length > 0:
            embedded_index = transitions.index(next_token)
            bit_string = bin(embedded_index)[2:].zfill(bit_length)
        else:
            bit_string = ""

        self.output += bit_string

    def solve(self):
        """Consumes the entire steganographic text and generates an output bitstream."""
        while not self.finished:
            self.step()

    def _get_transitions(self, gram):
        """Get possible transitions for the current gram in the Markov chain."""
        trans_matrix = self.model.chain.model[gram]
        trans_matrix = sorted(trans_matrix.items(), key=lambda kv: (kv[1]), reverse=True)
        transitions = [i[0] for i in trans_matrix]
        return transitions

    @staticmethod
    def _pretty_print_list(lst, limit):
        """Pretty print a list, showing only up to `limit` items."""
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
