import torch

def translate_sentence(sentence, encoder, decoder, source_vocab, target_vocab):
    encoder.eval()
    decoder.eval()

    # Convert sentence to tensor
    input_tensor = torch.tensor([source_vocab[token] for token in sentence.split()], dtype=torch.long).unsqueeze(1)

    # Get encoder outputs
    hidden, cell = encoder(input_tensor)

    # Decoder starts with <sos> token
    target_idx = target_vocab["<sos>"]
    output_sentence = []

    for _ in range(10):  # Max translation length
        target_tensor = torch.tensor([target_idx], dtype=torch.long)
        prediction, hidden, cell = decoder(target_tensor, hidden, cell)
        target_idx = prediction.argmax(1).item()
        if target_idx == target_vocab["<eos>"]:
            break
        output_sentence.append(target_idx)

    return [target_vocab.lookup_token(idx) for idx in output_sentence]

# Example usage
translated_text = translate_sentence("hello how are you", encoder, decoder, source_vocab, target_vocab)
print("Translated:", " ".join(translated_text))
