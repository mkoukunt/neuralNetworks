import random
import os
import re
import unicodedata
import zipfile

import requests
import torch
import torch.nn as nn
import torch.optim as optim
import tokenizers
import tqdm

#
# Data preparation
#

# Download dataset provided by Anki: https://www.manythings.org/anki/ with requests
if not os.path.exists("fra-eng.zip"):
    url = "http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip"
    response = requests.get(url)
    with open("fra-eng.zip", "wb") as f:
        f.write(response.content)


# Normalize text
# each line of the file is in the format "<english>\t<french>"
# We convert text to lowercasee, normalize unicode (UFKC)
def normalize(line):
    """Normalize a line of text and split into two at the tab character"""
    line = unicodedata.normalize("NFKC", line.strip().lower())
    eng, fra = line.split("\t")
    return eng.lower().strip(), fra.lower().strip()


text_pairs = []
with zipfile.ZipFile("fra-eng.zip", "r") as zip_ref:
    for line in zip_ref.read("fra.txt").decode("utf-8").splitlines():
        eng, fra = normalize(line)
        text_pairs.append((eng, fra))

#
# Tokenization with BPE
#

if os.path.exists("en_tokenizer.json") and os.path.exists("fr_tokenizer.json"):
    en_tokenizer = tokenizers.Tokenizer.from_file("en_tokenizer.json")
    fr_tokenizer = tokenizers.Tokenizer.from_file("fr_tokenizer.json")
else:
    en_tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
    fr_tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())

    # Configure pre-tokenizer to split on whitespace and punctuation, add space at beginning of the sentence
    en_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
    fr_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)

    # Configure decoder: So that word boundary symbol "Ġ" will be removed
    en_tokenizer.decoder = tokenizers.decoders.ByteLevel()
    fr_tokenizer.decoder = tokenizers.decoders.ByteLevel()

    # Train BPE for English and French using the same trainer
    VOCAB_SIZE = 8000
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[start]", "[end]", "[pad]"],
        show_progress=True
    )
    en_tokenizer.train_from_iterator([x[0] for x in text_pairs], trainer=trainer)
    fr_tokenizer.train_from_iterator([x[1] for x in text_pairs], trainer=trainer)

    en_tokenizer.enable_padding(pad_id=en_tokenizer.token_to_id("[pad]"), pad_token="[pad]")
    fr_tokenizer.enable_padding(pad_id=fr_tokenizer.token_to_id("[pad]"), pad_token="[pad]")

    # Save the trained tokenizers
    en_tokenizer.save("en_tokenizer.json", pretty=True)
    fr_tokenizer.save("fr_tokenizer.json", pretty=True)

# Test the tokenizer
print("Sample tokenization:")
en_sample, fr_sample = random.choice(text_pairs)
encoded = en_tokenizer.encode(en_sample)
print(f"Original: {en_sample}")
print(f"Tokens: {encoded.tokens}")
print(f"IDs: {encoded.ids}")
print(f"Decoded: {en_tokenizer.decode(encoded.ids)}")
print()

encoded = fr_tokenizer.encode("[start] " + fr_sample + " [end]")
print(f"Original: {fr_sample}")
print(f"Tokens: {encoded.tokens}")
print(f"IDs: {encoded.ids}")
print(f"Decoded: {fr_tokenizer.decode(encoded.ids)}")
print()


#
# Create PyTorch dataset for the BPE-encoded translation pairs
#

class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, text_pairs):
        self.text_pairs = text_pairs

    def __len__(self):
        return len(self.text_pairs)

    def __getitem__(self, idx):
        eng, fra = self.text_pairs[idx]
        return eng, "[start] " + fra + " [end]"


def collate_fn(batch):
    en_str, fr_str = zip(*batch)
    en_enc = en_tokenizer.encode_batch(en_str, add_special_tokens=True)
    fr_enc = fr_tokenizer.encode_batch(fr_str, add_special_tokens=True)
    en_ids = [enc.ids for enc in en_enc]
    fr_ids = [enc.ids for enc in fr_enc]
    return torch.tensor(en_ids), torch.tensor(fr_ids)


BATCH_SIZE = 32
dataset = TranslationDataset(text_pairs)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Test the dataset
for en_ids, fr_ids in dataloader:
    print(f"English: {en_ids}")
    print(f"French: {fr_ids}")
    break


#
# Create LSTM seq2seq model for translation
#

class EncoderLSTM(nn.Module):
    """A stacked LSTM encoder with an embedding layer"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        """
        Plain LSTM is used. No bidirectional LSTM.

        Args:
            vocab_size: The size of the input vocabulary
            embedding_dim: The dimension of the embedding vector
            hidden_dim: The dimension of the hidden state
            num_layers: The number of recurrent layers (layers of stacked LSTM)
            dropout: The dropout rate, applied to all LSTM layers except the last one
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, input_seq):
        # input seq = [batch_size, seq_len] -> embedded = [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(input_seq)
        # outputs = [batch_size, seq_len, embedding_dim]
        # hidden = cell = [n_layers, batch_size, hidden_dim]
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell


class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_seq, hidden, cell):
        # input seq = [batch_size, seq_len] -> embedded = [batch_size, seq_len, embedding_dim]
        # hidden = cell = [n_layers, batch_size, hidden_dim]
        embedded = self.embedding(input_seq)
        # output = [batch_size, seq_len, embedding_dim]
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.out(output)
        return prediction, hidden, cell


class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, target_seq):
        """Given the partial target sequence, predict the next token"""
        # input seq = [batch_size, seq_len]
        # target seq = [batch_size, seq_len]
        batch_size, target_len = target_seq.shape
        device = target_seq.device
        # storing output logits
        outputs = []
        # encoder forward pass
        _enc_out, hidden, cell = self.encoder(input_seq)
        dec_in = target_seq[:, :1]
        # decoder forward pass
        for t in range(target_len - 1):
            # last target token and hidden states -> next token
            pred, hidden, cell = self.decoder(dec_in, hidden, cell)
            # store the prediction
            pred = pred[:, -1:, :]
            outputs.append(pred)
            # use the predicted token as the next input
            dec_in = torch.cat([dec_in, pred.argmax(dim=2)], dim=1)
        outputs = torch.cat(outputs, dim=1)
        return outputs


# Initialize model parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
enc_vocab = len(en_tokenizer.get_vocab())
dec_vocab = len(fr_tokenizer.get_vocab())
emb_dim = 256
hidden_dim = 256
num_layers = 2
dropout = 0.1

# Create model
encoder = EncoderLSTM(enc_vocab, emb_dim, hidden_dim, num_layers, dropout).to(device)
decoder = DecoderLSTM(dec_vocab, emb_dim, hidden_dim, num_layers, dropout).to(device)
model = Seq2SeqLSTM(encoder, decoder).to(device)
print(model)

print("Model created with:")
print(f"  Input vocabulary size: {enc_vocab}")
print(f"  Output vocabulary size: {dec_vocab}")
print(f"  Embedding dimension: {emb_dim}")
print(f"  Hidden dimension: {hidden_dim}")
print(f"  Number of layers: {num_layers}")
print(f"  Dropout: {dropout}")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Train unless model.pth exists
if os.path.exists("seq2seq.pth"):
    model.load_state_dict(torch.load("seq2seq.pth"))
else:
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss(ignore_index=fr_tokenizer.token_to_id("[pad]"))
    N_EPOCHS = 30

    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0
        for en_ids, fr_ids in tqdm.tqdm(dataloader, desc="Training"):
            # Move the "sentences" to device
            en_ids = en_ids.to(device)
            fr_ids = fr_ids.to(device)
            # zero the grad, then forward pass
            optimizer.zero_grad()
            outputs = model(en_ids, fr_ids)
            # compute the loss: compare 3D logits to 2D targets
            loss = loss_fn(outputs.reshape(-1, dec_vocab), fr_ids[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{N_EPOCHS}; Avg loss {epoch_loss / len(dataloader)}; Latest loss {loss.item()}")
        torch.save(model.state_dict(), f"seq2seq-epoch-{epoch + 1}.pth")
        # Test
        if (epoch + 1) % 5 != 0:
            continue
        model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for en_ids, fr_ids in tqdm.tqdm(dataloader, desc="Evaluating"):
                en_ids = en_ids.to(device)
                fr_ids = fr_ids.to(device)
                outputs = model(en_ids, fr_ids)
                loss = loss_fn(outputs.reshape(-1, dec_vocab), fr_ids[:, 1:].reshape(-1))
                epoch_loss += loss.item()
        print(f"Eval loss: {epoch_loss / len(dataloader)}")

    # Save the final model
    torch.save(model.state_dict(), "seq2seq.pth")

# Test for a few samples
model.eval()
N_SAMPLES = 5
MAX_LEN = 60
with torch.no_grad():
    start_token = torch.tensor([fr_tokenizer.token_to_id("[start]")]).to(device)
    for en, true_fr in random.sample(text_pairs, N_SAMPLES):
        en_ids = torch.tensor(en_tokenizer.encode(en).ids).unsqueeze(0).to(device)
        _output, hidden, cell = model.encoder(en_ids)
        pred_ids = [start_token]
        for _ in range(MAX_LEN):
            decoder_input = torch.tensor(pred_ids).unsqueeze(0).to(device)
            output, hidden, cell = model.decoder(decoder_input, hidden, cell)
            output = output[:, -1, :].argmax(dim=1)
            pred_ids.append(output.item())
            # early stop if the predicted token is the end token
            if pred_ids[-1] == fr_tokenizer.token_to_id("[end]"):
                break
        # Decode the predicted IDs
        pred_fr = fr_tokenizer.decode(pred_ids)
        print(f"English: {en}")
        print(f"French: {true_fr}")
        print(f"Predicted: {pred_fr}")
        print()