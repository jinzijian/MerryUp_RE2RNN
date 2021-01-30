from model import baseModel
from torch import nn
import torch
import data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import utils


class trainer():
    def __init__(self, model, train_dataloader, test_dataloader, optimizer, epochs, word2idx, tag2idx, idx2word,
                 idx2tag, use_gpu):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.epochs = epochs
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.idx2word = idx2word
        self.idx2tag = idx2tag
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.model = self.model.cuda()

    def train(self):
        for i in range(self.epochs):
            avg_loss = 0
            for idx, batch in enumerate(self.train_dataloader):
                self.model.zero_grad()
                input = batch['input']
                label = batch['label']
                seqlen = batch['length']
                # 要把list of tensor 转化为tensor
                input = torch.vstack(input).transpose(0, 1)  # B * L tensor
                label = torch.vstack(label).transpose(0, 1)  # B * L tensor
                if self.use_gpu:
                    input = input.cuda()
                    label = label.cuda()
                output = self.model.forward(input, seqlen)  # B * L * C
                loss_fn = nn.CrossEntropyLoss()
                # cross entropy needs flatten
                output = utils.flatten(output, seqlen)
                label = utils.flatten(label, seqlen)
                loss = loss_fn(output, label)
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.item()
            avg_loss = avg_loss / len(self.train_dataloader)
            print('epoch %d' % i)
            print("loss is %f" % avg_loss)

            #  evaluate train result

