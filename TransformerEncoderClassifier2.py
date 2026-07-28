import math
import os
import torch
import torch.nn as nn
import tokenizers
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
df = pd.read_csv('data\\spam_ham.csv')
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
print(df_train.head())

label2id = {'ham': 0, 'spam': 1}
id2label = {i: label for label, i in label2id.items()}

TOKENIZER_PATH = 'spam_ham_tokenizer.json'
VOCAB_SIZE = 8000
MAX_LEN = 128

if os.path.exists(TOKENIZER_PATH):
    tokenizer = tokenizers.Tokenizer.from_file(TOKENIZER_PATH)
else:
    # WordLevel + whitespace/punctuation splitting is the closest equivalent of
    # torchtext's 'basic_english' tokenizer.
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(unk_token='[unk]'))
    tokenizer.normalizer = tokenizers.normalizers.Sequence([
        tokenizers.normalizers.NFD(),
        tokenizers.normalizers.Lowercase(),
        tokenizers.normalizers.StripAccents(),
    ])
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()

    trainer = tokenizers.trainers.WordLevelTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=['[unk]', '[pad]'],
        show_progress=True
    )
    # Only fit the vocab on the training split so the test split stays unseen.
    tokenizer.train_from_iterator(df_train['Message'].astype(str).tolist(), trainer)
    tokenizer.save(TOKENIZER_PATH, pretty=True)

# Pad to the longest sequence in each batch, so collate_fn gets a rectangular batch.
tokenizer.enable_truncation(max_length=MAX_LEN)
tokenizer.enable_padding(pad_id=tokenizer.token_to_id('[pad]'), pad_token='[pad]')

text = 'this is text'
print(tokenizer.encode(text).tokens)
hidden_size = 4


class TextDataset(Dataset):
    def __init__(self, df):
        self.texts = df['Message'].astype(str).tolist()
        self.labels = [label2id[label] for label in df['Category']]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def collate_fn(batch):
    texts, labels = zip(*batch)
    encodings = tokenizer.encode_batch(list(texts))
    sequences = torch.tensor([encoding.ids for encoding in encodings], dtype=torch.long)
    return sequences, torch.tensor(labels, dtype=torch.long)



class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Embeddings, self).__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, vocab_size=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

class SingleHeadAttention(nn.Module):
    def __init__(self, d_model, d_head_size):
        super().__init__()
        self.lin_key = nn.Linear(d_model, d_head_size, bias=False)
        self.lin_query = nn.Linear(d_model, d_head_size, bias=False)
        self.lin_value = nn.Linear(d_model, d_head_size, bias=False)
        self.d_model = d_model

    def forward(self, x):
        query = self.lin_query(x)
        key = self.lin_key(x)
        value = self.lin_value(x)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_model)
        p_attn = scores.softmax(dim=-1)
        x = torch.matmul(p_attn, value)

        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        d_k = d_model // h
        self.multi_head = nn.ModuleList([SingleHeadAttention(d_model, d_k) for _ in range(h)])
        self.lin_agg = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.multi_head], dim=-1)
        return self.lin_agg(x)

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        return self.dropout(self.norm(x1 + x2))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class SingleEncoder(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.res_1 = ResidualConnection(d_model, dropout)
        self.res_2 = ResidualConnection(d_model, dropout)

        self.d_model = d_model

    def forward(self, x):
        x_attn = self.self_attn(x)
        x_res_1 = self.res_1(x, x_attn)
        x_ff = self.feed_forward(x_res_1)
        x_res_2 = self.res_2(x_res_1, x_ff)

        return x_res_2

class EncoderBlocks(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(N)])
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class TransformerEncoderModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, d_ff, N,
                 dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "nheads must divide evenly into d_model"

        self.emb = Embeddings(d_model, vocab_size)
        self.pos_encoder = PositionalEncoding(d_model=d_model, vocab_size=vocab_size)

        attn = MultiHeadAttention(nhead, d_model)
        ff = FeedForward(d_model, d_ff, dropout)
        self.transformer_encoder = EncoderBlocks(SingleEncoder(d_model, attn, ff, dropout), N)
        self.classifier = nn.Linear(d_model, 2)
        self.d_model = d_model

    def forward(self, x):
        x = self.emb(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x



def train(model, dataset, epochs, lr, bs):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam((p for p in model.parameters()
      if p.requires_grad), lr=lr)
    train_dataset = TextDataset(dataset)
    # num_workers=0: the tokenizer used by collate_fn lives in this process, and
    # Windows' spawn start method would re-import this module in each worker.
    train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=bs, collate_fn=collate_fn, shuffle=True)

    # Training loop
    for epoch in range(epochs):
        total_loss_train = 0
        total_acc_train = 0
        for train_sequence, train_label in tqdm(train_dataloader):

            # Model prediction
            predictions = model(train_sequence.to(device))
            labels = train_label.to(device)
            loss = criterion(predictions, labels)

            # Calculate accuracy and loss per batch
            correct = predictions.argmax(axis=1) == labels
            acc = correct.sum().item() / correct.size(0)
            total_acc_train += correct.sum().item()
            total_loss_train += loss.item()

            # Backprop
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        print(f'Epochs: {epoch + 1} | Loss: {total_loss_train / len(train_dataset): .3f} | Accuracy: {total_acc_train / len(train_dataset): .3f}')

def predict(text):
  sequence = torch.tensor(tokenizer.encode(text).ids, dtype=torch.long).unsqueeze(0)
  model.eval()
  with torch.no_grad():
    output = model(sequence.to(device))
  prediction = id2label[output.argmax(axis=1).item()]

  return prediction

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerEncoderModel(tokenizer.get_vocab_size(), d_model=300, nhead=4, d_ff=50,
                                    N=6, dropout=0.1).to(device)

epochs = 15
lr = 1e-4
batch_size = 4
#train(model, df_train, epochs, lr, batch_size)

MODEL_PATH = 'spam_ham_classifier.pth'
#torch.save(model.state_dict(), MODEL_PATH)
print(f'saved {MODEL_PATH}')

# Rebuild the architecture with the same hyperparameters, then restore the weights.
model = TransformerEncoderModel(tokenizer.get_vocab_size(), d_model=300, nhead=4, d_ff=50,
                                    N=6, dropout=0.1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()
print(f'loaded {MODEL_PATH}')

print(predict('WINNER!! Claim your free prize now, call 09061701461'))
print(predict('ok lar, see you at lunch'))
print(predict('ok Madhu, you are fired'))
