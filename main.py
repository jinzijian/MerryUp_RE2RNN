from dataset import myDataSet
from torch.utils.data import DataLoader
import data
import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=16, type=int, help='should be int')
    parser.add_argument('--hidden_dim', default=100, type=int, help='should be int')
    parser.add_argument('--embedding_dim', default=100, type=int, help='should be int')
    parser.add_argument('--epochs', default=30, type=int, help='should be int')

    args = parser.parse_args()
    pass

    # Cuda
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using gpu")
    else:
        print('Using cpu')

    #  load data
    dir = '/p300/MerryUp_RE2RNN/dataset/ATIS/'
    train_tokens, train_tags, train_seqlen = data.load_data_ATIS(dir + 'train.json')
    test_tokens, test_tags, test_seqlen = data.load_data_ATIS(dir + 'test.json')
    # get all
    all_tokens, all_tags = data.get_all(train_tokens, test_tokens, train_tags, test_tags)
    #  get max
    max = data.get_maxlen(all_tokens)
    # get dict
    word2idx, idx2word, tag2idx, idx2tag = data.create_dict(all_tokens, all_tags)
    # add padding
    train_tokens = data.add_padding(train_tokens, max)
    test_tokens = data.add_padding(test_tokens, max)
    # convert2vec
    train_tokens, train_tags = data.convert2vec(train_tokens, train_tags, word2idx, tag2idx)
    test_tokens, test_tags = data.convert2vec(test_tokens, test_tags, word2idx=word2idx, tag2idx = tag2idx)
    # dataset
    train_dataset = myDataSet(train_tokens, train_tags, train_seqlen)
    test_dataset = myDataSet(test_tokens, test_tags, test_seqlen)
    # dataloader
    train_data = DataLoader(train_dataset, batch_size=args.batch_size)
    test_data = DataLoader(test_dataset, batch_size=args.batch_size)




