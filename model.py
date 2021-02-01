import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import load_embedding
import torch.nn.functional as F

class baseModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tag2idx, batch_size, use_gpu, idx2word, emb_path):
        super(baseModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tag2idx = tag2idx
        self.target_size = len(tag2idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True, dropout=0.5)
        self.hidden2tags = nn.Linear(hidden_dim, self.target_size)
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.idx2word = idx2word
        self.emb_path = emb_path
        # pretrain embeddings
        emb_vectors = load_embedding(self.emb_path, self.idx2word)
        self.embeds = nn.Embedding.from_pretrained(torch.from_numpy(emb_vectors).float(), freeze=True)

    def forward(self, sentence, length):
        embeds = self.embeds(sentence)  # B * L * D
        pack_padded_seq_input = pack_padded_sequence(embeds, length, batch_first=True, enforce_sorted=False)
        lstm_out, (hn, cn) = self.lstm(pack_padded_seq_input)  # B * L * H
        output_padded, output_lengths = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = self.hidden2tags(output_padded)  # B * L * C
        return lstm_out

    def predict(self, sentence, lengths):
        embeds = self.embeds(sentence)  # B * L * D
        pack_padded_seq_input = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, (hn, cn) = self.lstm(pack_padded_seq_input)  # B * L * H
        output_padded, output_lengths = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = self.hidden2tags(output_padded)  # B * L * C
        lstm_out = lstm_out[:, -1, :]
        lstm_out.squeeze(1)
        softmax_out = F.softmax(lstm_out[:, -1, :], dim=1)
        pred_label = torch.argmax(softmax_out, dim=1)
        return pred_label  # B * L
