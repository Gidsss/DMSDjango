import os
import markovify

def build_markov_json(filename: str, corpus_file: str, state_size: int = 2):
    """ Takes in a corpus txt file and constructs a Markov model in json"""
    with open(corpus_file, "r", encoding="utf-8", ) as f:  # Use UTF-8 encoding
       model = markovify.Text(f.read(), state_size=state_size)

    os.makedirs("markov_models", exist_ok=True)  # Ensure directory exists
    with open(f"markov_models/{filename}.json", "w", encoding="utf-8") as f:  # Use UTF-8 encoding for output
        f.write(model.to_json())

    print(f"Markov model has been saved as 'markov_models/{filename}.json'.")
     
if __name__ == "__main__":
    corpus_file = "clean_combined.txt"  
    model_filename = "legal_corpus" 
    
    # Build and save the Markov model
    build_markov_json(model_filename, corpus_file)