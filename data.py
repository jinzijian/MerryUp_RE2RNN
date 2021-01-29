import os
import pickle
import json


# Todo: convert data -> dataset

#load data
def load_data_ATIS(mode, DATA_DIR = '../data/ATIS'):
    query, slots, intent = pickle.load(open(os.path.join(DATA_DIR, 'atis.{}.new.pkl'.format(mode)), 'rb'))
    return query, slots, intent


def load_dict_ATIS(DATA_DIR = '../data/ATIS'):
    t2i, s2i, in2i, i2t, i2s, i2in, dicts = pickle.load(open(os.path.join(DATA_DIR, 'atis.{}.new.pkl'.format('dicts')), 'rb'))
    i2s = {k:v.lower() for k, v in i2s.items()}
    s2i = {k.lower():v for k, v in s2i.items()}
    i2t = {k:v.lower() for k, v in i2t.items()}
    t2i = {k.lower():v for k, v in t2i.items()}
    return t2i, i2t, s2i, i2s, dicts
