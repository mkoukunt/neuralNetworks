import random
import os
import re
import unicodedata
import zipfile

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    eng, fra = line.split(",")
    return eng.lower().strip(), fra.lower().strip()


text_pairs = []
#with zipfile.ZipFile("fra-eng.zip", "r") as zip_ref:
    #for line in zip_ref.read("fra.txt").decode("utf-8").splitlines():
        #eng, fra = normalize(line)
        #text_pairs.append((eng, fra))

#
# Tokenization with BPE
#
with open("data/ndp-cfg.txt", "r") as f:
    for line in f.read().splitlines():

        text_pairs.append(normalize(line))
        eng, fra = normalize(line)
        text_pairs.append((eng, fra))
if os.path.exists("en_tokenizer.json") and os.path.exists("fr_tokenizer.json"):
    en_tokenizer = tokenizers.Tokenizer.from_file("en_tokenizer.json")
    fr_tokenizer = tokenizers.Tokenizer.from_file("fr_tokenizer.json")
else:
    en_tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
    fr_tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())

    # Configure pre-tokenizer to split on whitespace and punctuation, add space at beginning of the sentence
    en_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
    fr_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True,use_regex=False)

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
    def __init__(self, text_pairs, en_tokenizer, fr_tokenizer):
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
dataset = TranslationDataset(text_pairs, en_tokenizer, fr_tokenizer)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Test the dataset
for en_ids, fr_ids in dataloader:
    print(f"English: {en_ids}")
    print(f"French: {fr_ids}")
    break


#
# Transformer model components
#

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len=1024):
        super().__init__()
        N = 10000
        inv_freq = 1. / (N ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(max_seq_len).float()
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        sinusoid_inp = torch.outer(position, inv_freq)
        self.register_buffer("cos", sinusoid_inp.cos())
        self.register_buffer("sin", sinusoid_inp.sin())

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(1)
        cos = self.cos[:seq_len].view(1, seq_len, 1, -1)
        sin = self.sin[:seq_len].view(1, seq_len, 1, -1)
        return apply_rotary_pos_emb(x, cos, sin)


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, intermediate_dim)
        self.up = nn.Linear(hidden_dim, intermediate_dim)
        self.down = nn.Linear(intermediate_dim, hidden_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.gate(x)) * self.up(x)
        x = self.down(x)
        return x


class GQA(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads=None, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_groups = num_heads // num_kv_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, q, k, v, mask=None, rope=None):
        q_batch_size, q_seq_len, hidden_dim = q.shape
        k_batch_size, k_seq_len, hidden_dim = k.shape
        v_batch_size, v_seq_len, hidden_dim = v.shape

        # projection
        q = self.q_proj(q).view(q_batch_size, q_seq_len, -1, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(k_batch_size, k_seq_len, -1, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(v_batch_size, v_seq_len, -1, self.head_dim).transpose(1, 2)

        # apply rotary positional encoding
        if rope:
            q = rope(q)
            k = rope(k)

        # compute grouped query attention
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        output = F.scaled_dot_product_attention(q, k, v,
                                                attn_mask=mask,
                                                dropout_p=self.dropout,
                                                enable_gqa=True)
        output = output.transpose(1, 2).reshape(q_batch_size, q_seq_len, hidden_dim).contiguous()
        output = self.out_proj(output)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads=None, dropout=0.1):
        super().__init__()
        self.self_attn = GQA(hidden_dim, num_heads, num_kv_heads, dropout)
        self.mlp = SwiGLU(hidden_dim, 4 * hidden_dim)
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)

    def forward(self, x, mask=None, rope=None):
        # self-attention sublayer
        out = x
        out = self.norm1(x)
        out = self.self_attn(out, out, out, mask, rope)
        x = out + x
        # MLP sublayer
        out = self.norm2(x)
        out = self.mlp(out)
        return out + x


class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads=None, dropout=0.1):
        super().__init__()
        self.self_attn = GQA(hidden_dim, num_heads, num_kv_heads, dropout)
        self.cross_attn = GQA(hidden_dim, num_heads, num_kv_heads, dropout)
        self.mlp = SwiGLU(hidden_dim, 4 * hidden_dim)
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)
        self.norm3 = nn.RMSNorm(hidden_dim)

    def forward(self, x, enc_out, mask=None, rope=None):
        # self-attention sublayer
        out = x
        out = self.norm1(out)
        out = self.self_attn(out, out, out, mask, rope)
        x = out + x
        # cross-attention sublayer
        out = self.norm2(x)
        out = self.cross_attn(out, enc_out, enc_out, None, rope)
        x = out + x
        # MLP sublayer
        x = out + x
        out = self.norm3(x)
        out = self.mlp(out)
        return out + x


class Transformer(nn.Module):
    def __init__(self, num_layers, num_heads, num_kv_heads, hidden_dim,
                 max_seq_len, vocab_size_src, vocab_size_tgt, dropout=0.1):
        super().__init__()
        self.rope = RotaryPositionalEncoding(hidden_dim // num_heads, max_seq_len)
        self.src_embedding = nn.Embedding(vocab_size_src, hidden_dim)
        self.tgt_embedding = nn.Embedding(vocab_size_tgt, hidden_dim)
        self.encoders = nn.ModuleList([
            EncoderLayer(hidden_dim, num_heads, num_kv_heads, dropout) for _ in range(num_layers)
        ])
        self.decoders = nn.ModuleList([
            DecoderLayer(hidden_dim, num_heads, num_kv_heads, dropout) for _ in range(num_layers)
        ])
        self.out = nn.Linear(hidden_dim, vocab_size_tgt)

    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None):
        # Encoder
        x = self.src_embedding(src_ids)
        for encoder in self.encoders:
            x = encoder(x, src_mask, self.rope)
        enc_out = x
        # Decoder
        x = self.tgt_embedding(tgt_ids)
        for decoder in self.decoders:
            x = decoder(x, enc_out, tgt_mask, self.rope)
        return self.out(x)


model_config = {
    "num_layers": 4,
    "num_heads": 8,
    "num_kv_heads": 4,
    "hidden_dim": 128,
    "max_seq_len": 768,
    "vocab_size_src": len(en_tokenizer.get_vocab()),
    "vocab_size_tgt": len(fr_tokenizer.get_vocab()),
    "dropout": 0.1,
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Transformer(**model_config).to(device)
print(model)

# Training

print("Model created with:")
print(f"  Input vocabulary size: {model_config['vocab_size_src']}")
print(f"  Output vocabulary size: {model_config['vocab_size_tgt']}")
print(f"  Number of layers: {model_config['num_layers']}")
print(f"  Number of heads: {model_config['num_heads']}")
print(f"  Number of KV heads: {model_config['num_kv_heads']}")
print(f"  Hidden dimension: {model_config['hidden_dim']}")
print(f"  Max sequence length: {model_config['max_seq_len']}")
print(f"  Dropout: {model_config['dropout']}")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


def create_causal_mask(seq_len, device):
    """
    Create a causal mask for autoregressive attention.

    Args:
        seq_len: Length of the sequence

    Returns:
        Causal mask of shape (seq_len, seq_len)
    """
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
    return mask


def create_padding_mask(batch, padding_token_id):
    """
    Create a padding mask for a batch of sequences.

    Args:
        batch: Batch of sequences, shape (batch_size, seq_len)
        padding_token_id: ID of the padding token

    Returns:
        Padding mask of shape (batch_size, seq_len, seq_len)
    """
    batch_size, seq_len = batch.shape
    device = batch.device
    padded = torch.zeros_like(batch, device=device).float().masked_fill(batch == padding_token_id, float('-inf'))
    mask = torch.zeros(batch_size, seq_len, seq_len, device=device) + padded[:, :, None] + padded[:, None, :]
    return mask[:, None, :, :]


# Train unless model.pth exists
loss_fn = nn.CrossEntropyLoss(ignore_index=fr_tokenizer.token_to_id("[pad]"))
if os.path.exists("transformer.pth"):
    model.load_state_dict(torch.load("transformer.pth"))
else:
    N_EPOCHS = 10
    LR = 0.005
    WARMUP_STEPS = 1000
    CLIP_NORM = 5.0
    best_loss = float('inf')
    optimizer = optim.Adam(model.parameters(), lr=LR)
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                                                   total_iters=WARMUP_STEPS)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS * len(dataloader) - WARMUP_STEPS,
                                                            eta_min=0)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                                milestones=[WARMUP_STEPS])
    print(f"Training for {N_EPOCHS} epochs with {len(dataloader)} steps per epoch")

    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0
        for en_ids, fr_ids in tqdm.tqdm(dataloader, desc="Training"):
            # Move the "sentences" to device
            en_ids = en_ids.to(device)
            fr_ids = fr_ids.to(device)
            # create source mask as padding mask, target mask as causal mask
            src_mask = create_padding_mask(en_ids, en_tokenizer.token_to_id("[pad]"))
            tgt_mask = create_causal_mask(fr_ids.shape[1], device).unsqueeze(0) + create_padding_mask(fr_ids,
                                                                                                      fr_tokenizer.token_to_id(
                                                                                                          "[pad]"))
            # zero the grad, then forward pass
            optimizer.zero_grad()
            outputs = model(en_ids, fr_ids, src_mask, tgt_mask)
            # compute the loss: compare 3D logits to 2D targets
            loss = loss_fn(outputs[:, :-1, :].reshape(-1, outputs.shape[-1]), fr_ids[:, 1:].reshape(-1))
            loss.backward()
            if CLIP_NORM:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM, error_if_nonfinite=False)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{N_EPOCHS}; Avg loss {epoch_loss / len(dataloader)}; Latest loss {loss.item()}")
        # Test
        model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for en_ids, fr_ids in tqdm.tqdm(dataloader, desc="Evaluating"):
                en_ids = en_ids.to(device)
                fr_ids = fr_ids.to(device)
                src_mask = create_padding_mask(en_ids, en_tokenizer.token_to_id("[pad]"))
                tgt_mask = create_causal_mask(fr_ids.shape[1], device).unsqueeze(0) + create_padding_mask(fr_ids,
                                                                                                          fr_tokenizer.token_to_id(
                                                                                                              "[pad]"))
                outputs = model(en_ids, fr_ids, src_mask, tgt_mask)
                loss = loss_fn(outputs[:, :-1, :].reshape(-1, outputs.shape[-1]), fr_ids[:, 1:].reshape(-1))
                epoch_loss += loss.item()
        print(f"Eval loss: {epoch_loss / len(dataloader)}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), f"transformer-epoch-{epoch + 1}.pth")

    # Save the final model after training
    torch.save(model.state_dict(), "transformer.pth")

# Test for a few samples
model.eval()
N_SAMPLES = 5
MAX_LEN = 60
with torch.no_grad():
    start_token = torch.tensor([fr_tokenizer.token_to_id("[start]")]).to(device)
    for en, true_fr in random.sample(dataset.text_pairs, N_SAMPLES):
        en_ids = torch.tensor(en_tokenizer.encode("device 7 shared line no").ids).unsqueeze(0).to(device)

        # get context from encoder
        src_mask = create_padding_mask(en_ids, en_tokenizer.token_to_id("[pad]"))
        x = model.src_embedding(en_ids)
        for encoder in model.encoders:
            x = encoder(x, src_mask, model.rope)
        enc_out = x

        # generate output from decoder
        fr_ids = start_token.unsqueeze(0)
        for _ in range(MAX_LEN):
            tgt_mask = create_causal_mask(fr_ids.shape[1], device).unsqueeze(0)
            tgt_mask = tgt_mask + create_padding_mask(fr_ids, fr_tokenizer.token_to_id("[pad]"))
            x = model.tgt_embedding(fr_ids)
            for decoder in model.decoders:
                x = decoder(x, enc_out, tgt_mask, model.rope)
            outputs = model.out(x)

            outputs = outputs.argmax(dim=-1)
            fr_ids = torch.cat([fr_ids, outputs[:, -1:]], axis=-1)
            if fr_ids[0, -1] == fr_tokenizer.token_to_id("[end]"):
                break

        # Decode the predicted IDs
        pred_fr = fr_tokenizer.decode(fr_ids[0].tolist())
        print(f"English: {en}")
        print(f"French: {true_fr}")
        print(f"Predicted: {pred_fr}")
        print()

