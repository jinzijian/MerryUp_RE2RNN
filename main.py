#
import argparse
import json
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=20, type=int, help='should be int')
    parser.add_argument('--hidden_dim', default=150, type=int, help='should be int')
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
