import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

def yield_tokens(data):
    for sentence in data:
        yield sentence.split()

# Example dataset
source_sentences = ["hello how are you", "I love machine learning"]
target_sentences = ["hola cómo estás", "me encanta el aprendizaje automático"]

# Tokenizer
tokenizer = get_tokenizer("basic_english")

# Vocabularies
source_vocab = build_vocab_from_iterator(yield_tokens(source_sentences), specials=["<unk>", "<pad>", "<sos>", "<eos>"])
target_vocab = build_vocab_from_iterator(yield_tokens(target_sentences), specials=["<unk>", "<pad>", "<sos>", "<eos>"])

# Convert sentences to tensors
def sentence_to_tensor(sentence, vocab):
    return torch.tensor([vocab[token] for token in sentence.split()], dtype=torch.long)

source_tensors = [sentence_to_tensor(s, source_vocab) for s in source_sentences]
target_tensors = [sentence_to_tensor(t, target_vocab) for t in target_sentences]

print("Sample source tensor:", source_tensors[0])
