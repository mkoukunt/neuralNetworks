"""Causal (decoder-only) transformer trained from scratch for next-word prediction.

Tokenization uses the HuggingFace `tokenizers` library, same as seq2seq.py.
The corpus is the SMS text in data/spam_ham.csv, packed into a single stream of
token ids and chunked into fixed-length blocks -- so every position in every
block is a training example, and no padding is needed.
"""

import math
import os

import pandas as pd
import tokenizers
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Data / tokenizer.
CORPUS_PATH = 'data\\spam_ham.csv'
TOKENIZER_PATH = 'causal_lm_tokenizer.json'
CHECKPOINT_PATH = 'causal_transformer.pth'
VOCAB_SIZE = 8000

# Model.
BLOCK_SIZE = 64      # context length in tokens
D_MODEL = 256
N_HEAD = 4
N_LAYER = 4
D_FF = 1024
DROPOUT = 0.1

# Training.
EPOCHS = 5
LR = 3e-4
BATCH_SIZE = 32
VAL_FRACTION = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_tokenizer(texts):
    """Load the trained tokenizer, or train a byte-level BPE on `texts`."""
    if os.path.exists(TOKENIZER_PATH):
        return tokenizers.Tokenizer.from_file(TOKENIZER_PATH)

    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token='[unk]'))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
    # Removes the "Ġ" word-boundary marker when decoding generated text.
    tokenizer.decoder = tokenizers.decoders.ByteLevel()

    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=['[unk]', '[eos]'],
        show_progress=True
    )
    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.save(TOKENIZER_PATH, pretty=True)
    return tokenizer


def build_id_stream(tokenizer, texts):
    """Tokenize every document and concatenate into one flat tensor of ids."""
    eos_id = tokenizer.token_to_id('[eos]')
    ids = []
    for encoding in tokenizer.encode_batch(texts):
        ids.extend(encoding.ids)
        ids.append(eos_id)
    return torch.tensor(ids, dtype=torch.long)


class NextTokenDataset(Dataset):
    """Chunks a flat id stream into (input, target) pairs shifted by one token."""

    def __init__(self, ids, block_size):
        self.ids = ids
        self.block_size = block_size

    def __len__(self):
        # -1 because the last block needs one extra token to build its target.
        return max(0, (len(self.ids) - 1) // self.block_size)

    def __getitem__(self, idx):
        start = idx * self.block_size
        chunk = self.ids[start:start + self.block_size + 1]
        return chunk[:-1], chunk[1:]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where position t may only attend to <= t."""

    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0, 'n_head must divide evenly into d_model'
        self.n_head = n_head
        self.d_head = d_model // n_head

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask):
        batch, seq_len, d_model = x.size()

        query, key, value = self.qkv(x).split(d_model, dim=-1)
        # (batch, n_head, seq_len, d_head)
        query = query.view(batch, seq_len, self.n_head, self.d_head).transpose(1, 2)
        key = key.view(batch, seq_len, self.n_head, self.d_head).transpose(1, 2)
        value = value.view(batch, seq_len, self.n_head, self.d_head).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores.masked_fill(causal_mask[:, :, :seq_len, :seq_len], float('-inf'))
        p_attn = self.attn_dropout(scores.softmax(dim=-1))

        out = torch.matmul(p_attn, value)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.resid_dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w_2(self.w_1(x).relu()))


class DecoderBlock(nn.Module):
    """Pre-norm block: normalize before each sublayer, add the residual after."""

    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.self_attn = CausalSelfAttention(d_model, n_head, dropout)
        self.norm_2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, causal_mask):
        x = x + self.self_attn(self.norm_1(x), causal_mask)
        x = x + self.feed_forward(self.norm_2(x))
        return x


class CausalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=D_MODEL, n_head=N_HEAD, n_layer=N_LAYER,
                 d_ff=D_FF, block_size=BLOCK_SIZE, dropout=DROPOUT):
        super().__init__()
        self.block_size = block_size
        self.d_model = d_model

        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, block_size, dropout)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, n_head, d_ff, dropout) for _ in range(n_layer)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying: the output projection reuses the embedding table.
        self.lm_head.weight = self.emb.weight

        # True marks positions that must not be attended to (strictly future).
        mask = torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1)
        self.register_buffer('causal_mask', mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        assert x.size(1) <= self.block_size, \
            f'sequence length {x.size(1)} exceeds block size {self.block_size}'

        x = self.emb(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        for block in self.blocks:
            x = block(x, self.causal_mask)
        x = self.norm(x)
        return self.lm_head(x)


def run_epoch(model, dataloader, criterion, optimizer=None):
    """Runs one pass; trains when an optimizer is given, evaluates otherwise."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.set_grad_enabled(is_train):
        for inputs, targets in tqdm(dataloader, leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

            n_tokens = targets.numel()
            total_loss += loss.item() * n_tokens
            total_correct += (logits.argmax(dim=-1) == targets).sum().item()
            total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    return avg_loss, total_correct / total_tokens, math.exp(min(avg_loss, 20))


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=30, temperature=1.0, top_k=20):
    """Autoregressively samples a continuation of `prompt`."""
    model.eval()
    eos_id = tokenizer.token_to_id('[eos]')
    ids = tokenizer.encode(prompt).ids

    for _ in range(max_new_tokens):
        # Feed only the most recent block_size tokens of context.
        context = torch.tensor([ids[-model.block_size:]], dtype=torch.long, device=device)
        logits = model(context)[:, -1, :] / max(temperature, 1e-6)

        if top_k:
            kth_value = torch.topk(logits, min(top_k, logits.size(-1)))[0][:, -1:]
            logits = logits.masked_fill(logits < kth_value, float('-inf'))

        next_id = torch.multinomial(logits.softmax(dim=-1), num_samples=1).item()
        if next_id == eos_id:
            break
        ids.append(next_id)

    return tokenizer.decode(ids)


if __name__ == '__main__':
    print('device:', device)

    texts = pd.read_csv(CORPUS_PATH)['Message'].astype(str).tolist()
    tokenizer = build_tokenizer(texts)
    vocab_size = tokenizer.get_vocab_size()

    ids = build_id_stream(tokenizer, texts)
    # Split by position in the stream so train and validation text never overlap.
    n_val = int(len(ids) * VAL_FRACTION)
    train_dataset = NextTokenDataset(ids[:len(ids) - n_val], BLOCK_SIZE)
    val_dataset = NextTokenDataset(ids[len(ids) - n_val:], BLOCK_SIZE)
    print(f'{len(ids):,} tokens | vocab {vocab_size:,} | '
          f'{len(train_dataset):,} train blocks | {len(val_dataset):,} val blocks')

    # num_workers=0: on Windows the spawn start method re-imports this module.
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = CausalTransformer(vocab_size).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} parameters')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        train_loss, train_acc, train_ppl = run_epoch(model, train_dataloader, criterion, optimizer)
        val_loss, val_acc, val_ppl = run_epoch(model, val_dataloader, criterion)
        print(f'Epoch {epoch + 1}/{EPOCHS} | '
              f'train loss {train_loss:.3f} acc {train_acc:.3f} ppl {train_ppl:.1f} | '
              f'val loss {val_loss:.3f} acc {val_acc:.3f} ppl {val_ppl:.1f}')

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f'saved {CHECKPOINT_PATH}')

    for prompt in ['i will call you', 'you have won a', 'sorry i']:
        print(f'{prompt!r} -> {generate(model, tokenizer, prompt)!r}')
