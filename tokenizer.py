class ToyCharTokenizer:
    """A simple character-level tokenizer.

    Maps unique characters in a text corpus to integers and back.
    """
    def __init__(self, text: str):
        # Extract unique characters and sort them for deterministic indexing
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        # String to Integer
        self.stoi = {ch: i for i, ch in enumerate(chars)}

        # Integer to String
        self.itos = {i: ch for i, ch in enumerate(chars)}

        print(
            f"Initialized ToyCharTokenizer. Vocabulary size (unique chars):"
            f" {self.vocab_size}"
        )

    def encode(self, s:str) -> list[int]:
        """Converts a string of text into a list of integer token IDs."""
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, l: list[int]) -> str:
        """Converts a list of integer token IDs back into a readable string."""
        return "".join([self.itos[i] for i in l])

if __name__ == "__main__":
    sample_text = (
        "Building a small language model from scratch is awesome!"
    )

    tokenizer = ToyCharTokenizer(sample_text)

    encoded_tokens = tokenizer.encode(sample_text)
    print(f"\nOriginal text: {sample_text}")
    print(f"Encoded IDs (first 15): {encoded_tokens[:15]}...")

    decoded_text = tokenizer.decode(encoded_tokens)
    print(f"Decoded text: {decoded_text}")

