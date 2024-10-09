import os
import time
import markovify
import random
from math import ceil, log2


def build_markov_json(filename: str, corpus_file: str, state_size: int = 2):
	""" Takes in a corpus txt file and constructs a markov model in json """
	with open(corpus_file, "r", encoding="utf8") as f:
		model = markovify.Text(f.read(), state_size=state_size)
	os.makedirs("markov_models", exist_ok=True)
	with open(f"markov_models/{filename}.json", "w") as f:
		f.write(model.to_json())


def build_model(json_file: str):
	""" Takes in a json markov model and builds it. Returns the constructed Markov Model. """
	with open(json_file, "r") as f:
		return markovify.Text.from_json(f.read())


def _string_to_bitstream(data_string):
	byte_array = bytearray(data_string, "ascii")

	# Convert each byte to a binary string.
	bit_list = []
	for byte in byte_array:
		# Left pad the binary string with 0s to make it 8 bits long.
		binary_string = bin(byte)[2:].zfill(8)
		# Append the binary string to the bit list.
		bit_list.extend([str(int(bit)) for bit in binary_string])

	return "".join(bit_list)


class Encoder:
	"""
	Encodes a bitstream using a Markov Model.
	For basic usage, run `self.generate()` and get the generated output from `self.output_str`.
	"""

	def __init__(self, model: markovify.Text, bitstream: str, logging: bool):
		"""
		Initializes a Markov Encoder.

		Parameters:
			model (markovify.Text): The markov chain model to use in encoding
			bitstream (str): The bitstream to encode
			logging (bool): Optional parameter to enable logging
		"""
		self.model = model
		self.bitstream = bitstream

		if self.model.state_size == 1:
			self.entrypoints = [key for key in model.chain.model.get(("___BEGIN__",)).keys()]
		else:
			self.entrypoints = [key[-1] for key in model.chain.model.keys()
								if key.count("___BEGIN__") == self.model.state_size - 1][1:]

		self.current_gram = None
		self.output_tokens = []
		self.exhausts = 0
		self.end_key = 0
		self.exhausted = True
		self.finished = False

		self.logging = logging

	@property
	def output(self):
		""" Returns the current state of the output string. """
		return " ".join(self.output_tokens)

	def step(self):
		""" Generates a new word for the output and appends it to the output string. """
		if self.finished:
			return

		# Display limits for logging
		char_limit = 20
		matrix_limit = 10

		# Choose new starting point
		if self.exhausted:
			self.exhausted = False
			next_token, removed, bit_length, encoded_index = self._consume_from_list(self.entrypoints)

			if self.logging:
				if len(self.bitstream) > char_limit:
					remaining = f"{self.bitstream[:char_limit]}..."
				else:
					remaining = self.bitstream

				print(f"Entrypoint Chosen: {next_token}")
				print(f"\tMatrix Length: {len(self.entrypoints)}\tMax Bit Length: {bit_length}")
				print(f"\tBitstream: [{removed}]-{remaining}")
				print(f"\tEncoded Index: {encoded_index}\tToken: {next_token}")
				print()

			# Construct gram
			if self.model.state_size == 1:
				self.current_gram = (next_token,)
			else:
				self.current_gram = (*["___BEGIN__"]*(self.model.state_size-1), next_token)

		# Get next word
		else:
			transitions = self._get_transitions(self.current_gram)
			if "___END__" in transitions:
				if self.logging:
					if len(self.bitstream) > char_limit:
						remaining = f"{self.bitstream[:char_limit]}..."
					else:
						remaining = self.bitstream
					print(f"{self.current_gram}:")
					print(f"\tPossible Transitions: {self._pretty_print_list(transitions, matrix_limit)}")
					print(f"\tBitstream: []-{remaining}")
					print("\t-- Exhausting --")
					print()

				self.exhausted = True
				self.current_gram = None
				return

			next_token, removed, bit_length, encoded_index = self._consume_from_list(transitions)

			if self.logging:
				if len(self.bitstream) > char_limit:
					remaining = f"{self.bitstream[:char_limit]}..."
				else:
					remaining = self.bitstream
				print(f"{self.current_gram}:")
				print(f"\tPossible Transitions: {self._pretty_print_list(transitions, matrix_limit)}")
				print(f"\tMatrix Length: {len(transitions)}\tMax Bit Length: {bit_length}")
				print(f"\tBitstream: [{removed}]-{remaining}")
				print(f"\tEncoded Index: {encoded_index}\tToken: {next_token}")
				print()

			# Construct gram
			next_gram = list(self.current_gram)
			next_gram.append(next_token)
			self.current_gram = tuple(next_gram[1:])

		# Add token to output
		if type(next_token) == tuple:
			self.output_tokens.extend(next_token)
		else:
			self.output_tokens.append(next_token)

		if not self.bitstream:
			self.end_key = len(removed)

			# Inject end key into output
			i = random.randint(0, len(self.output_tokens) - 1)
			char_key = chr(self.end_key + 97)
			self.output_tokens[i] += char_key

			if self.logging:
				os.system("")
				injected_word = self.output_tokens[i]
				print(f"Output: {self.output}")
				print(f"\tEnd Key: {self.end_key} ({char_key})")
				print(f"\tInjection Point: \"{injected_word[:-1]}\" at index {i}")
				print(f"\tInjection Preview: ... {' '.join(self.output_tokens[max(0, i - 2):i])} "
					  f"{f'|{injected_word}|'} "
					  f"{' '.join(self.output_tokens[i + 1: min(len(self.output_tokens) - 1, i + 3)])} ...")

			self.finished = True

	def generate(self):
		""" Consumes the entire bitstream and generates the output for it """

		while not self.finished:
			self.step()

		return self.output

	def _consume_from_list(self, lst):
		# Get max possible bit length based on length of list
		list_length = len(lst)
		bit_length = ceil(log2(list_length))
		if list_length < 2 ** bit_length:
			bit_length -= 1

		# Read bit stream to get index
		encoded_index = 0 if bit_length == 0 else int(self.bitstream[:bit_length], 2)

		# Get next token based on index
		next_token = lst[encoded_index]

		# Log
		removed = self.bitstream[:bit_length]
		self.bitstream = self.bitstream[bit_length:]

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

	def __init__(self, model: markovify.Text, stega_text: str, logging: bool):
		"""
		Initializes a Markov Decoder. Run `self.solve()` and get the generated output from `self.output`.

		Parameters:
			model (markovify.Text): The markov chain model to use in decoding
			stega_text (str): The steganographic text to decode
			logging (bool): Optional parameter to enable logging
		"""
		self.model = model
		self.stega_text = stega_text.split(" ")

		if self.model.state_size == 1:
			self.entrypoints = [key for key in model.chain.model.get(("___BEGIN__",)).keys()]
		else:
			self.entrypoints = [key[-1] for key in model.chain.model.keys()
								if key.count("___BEGIN__") == self.model.state_size - 1][1:]
		self.endkey = 0
		self.current_gram = None
		self.index = 0
		self.exhausted = True
		self.finished = False
		self.logging = logging

		self.output = ""

	def step(self):
		""" Consumes a word from the steganographic text and appends the appropriate bits to the output """

		# Display limits for logging
		matrix_limit = 10
		char_limit = 20

		# Finish if index is at the end of the stega text
		if self.index >= len(self.stega_text) - 1 and not self.exhausted:
			if self.logging:
				print(f"\nOutput:\t\t{self.output}")
			self.finished = True
			return

		previous = self.output if len(self.output) <= char_limit else f"...{self.output[-char_limit:]}"
		if self.exhausted:
			token = self.stega_text[self.index]

			# Check for end key
			if token not in self.entrypoints:
				key_char = token[-1]
				token = token[:-1]
				self.endkey = ord(key_char) - 97

				if self.logging:
					print(f"-- END KEY FOUND --")
					print(f"\tToken: {token}({key_char})")
					print(f"\tEncoded End Key: {self.endkey}")
					print()

			if self.model.state_size == 1:
				self.current_gram = (token,)
			else:
				self.current_gram = (*["___BEGIN__"]*(self.model.state_size-1), token)

			embedded_index = self.entrypoints.index(token)
			bit_length = ceil(log2(len(self.entrypoints)))
			if len(self.entrypoints) < 2 ** bit_length:
				bit_length -= 1
			bit_length = self.endkey if self.index == len(self.stega_text) - 1 else bit_length
			bit_string = bin(embedded_index)[2:].zfill(bit_length)

			if self.logging:
				print(f"Entrypoint: {self.current_gram[1]}")
				print(f"\tIndex: {embedded_index} ({bit_string})")
				print(f"\tBitstream: {previous}-[{bit_string}]")
				print()

			# if "___END__" in self.get_transitions(self.current_gram):
			# 	self.index += 1
			# 	self.exhausted = True
			# else:
			# 	self.exhausted = False
			self.exhausted = False
		else:
			transitions = self._get_transitions(self.current_gram)
			at_end = self.index == len(self.stega_text) - 1

			# Get max possible bit length based on length of list
			list_length = len(transitions)
			bit_length = ceil(log2(list_length))
			if list_length < 2 ** bit_length:
				bit_length -= 1
			bit_length = 0 if list_length == 1 else bit_length

			if "___END__" in transitions:
				if self.logging:
					print(f"{self.current_gram}:")
					print(f"\tPossible Transitions: {self._pretty_print_list(transitions, matrix_limit)}")
					print(f"\tBitstream: {previous}-[]")
					print("\t-- Exhausting --")
					print()

				self.exhausted = True
				self.current_gram = None
				self.index += 1
				return
			else:
				next_token = "" if at_end else self.stega_text[self.index + 1]

			# Check for end key
			if next_token not in transitions and not at_end:
				key_char = next_token[-1]
				self.endkey = ord(key_char) - 97
				next_token = next_token[:-1]

				if self.logging:
					print(f"-- END KEY FOUND --")
					print(f"\tToken: {next_token}({key_char})")
					print(f"\tEncoded End Key: {self.endkey}")
					print()

			bit_length = self.endkey if self.index == len(self.stega_text) - 2 else bit_length

			if bit_length != 0:
				embedded_index = "N/A" if at_end else transitions.index(next_token)
				bit_string = "" if at_end else bin(embedded_index)[2:].zfill(bit_length)
			else:
				embedded_index = "N/A"
				bit_string = ""

			if self.logging:
				print(f"{self.current_gram}:")
				print(f"\tPossible Transitions: {self._pretty_print_list(transitions, matrix_limit)}")
				print(f"\tNext Token: {next_token}\tIndex: {embedded_index} {f'({bit_string})' if bit_string else ''}")
				print(f"\tBitstream: {previous}-[{bit_string}]")
				print()

			# Construct gram
			next_gram = list(self.current_gram)
			next_gram.append(next_token)
			self.current_gram = tuple(next_gram[1:])

			self.index += 1

		# Add bit string to output
		self.output += bit_string

	def solve(self):
		""" Consumes the entire steganographic text and generates an output bitstream """
		while not self.finished:
			self.step()

		return self.output

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


# For quick testing
if __name__ == '__main__':
	print("Building models...")
	model_state_1 = build_model("markov_models/model_state_1.json")
	model_state_2 = build_model("markov_models/model_state_2.json")
	model_state_3 = build_model("markov_models/model_state_3.json")

	models = [model_state_1, model_state_2, model_state_3]

	text = "acehappybirthday071603!"
	bitstream = _string_to_bitstream(text)

	encoders = [Encoder(model, bitstream, False) for model in models]
	outputs = []

	print(f"\nText to be Encoded: {text}")
	for i, encoder in enumerate(encoders):
		print(f"\n-- Encoding with State Size {i + 1} --")
		a = time.perf_counter()
		output = encoder.generate()
		b = time.perf_counter()
		outputs.append(output)
		print(f"Output: {output}")
		print(f"Statistics: ")
		print(f"\t- Speed: {b - a: 0.4f} secs")
		print(f"\t- Embedding Rate: {len(bitstream) * 100 / (len(output) * 8): 0.2f}%")

	decoders = [Decoder(model, output, False) for model, output in zip(models, outputs)]
	print("\nTesting decoding...")
	for i, decoder in enumerate(decoders):
		print(f"\n-- Decoding with State Size {i + 1} --")
		a = time.perf_counter()
		output = decoder.solve()
		b = time.perf_counter()
		print(f"Output: {output}")
		print(f"Statistics: ")
		print(f"\t- Speed: {b - a: 0.4f} secs")
		print(f"\t- Valid: {output == bitstream}")
