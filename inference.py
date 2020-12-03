# -*- coding:utf-8 -*-

from interest_rec_gnn import *
import os
import sys
import pandas as pd
import numpy as np
import pickle
from sys import argv


def pickle_load(fpath):
    with open(fpath,'rb') as f:
        h = pickle.load(f)
    return h

def interest_model_inference(test_link_feats,link2id,domain2id,interest_feats,interest_graph,index2interests,model_states):
    link_ids = [commonlink2index_(i[0], link2id) for i in test_link_feats]
    domain_ids = [commondomain2index_(extract_domain_name(i[0]), domain2id) for i in test_link_feats]
    link_text = [process_text(i[1]) for i in test_link_feats]
    testset = list(zip(link_ids, link_text, domain_ids))
    interest_model = Interest_Rec_Model(testset, interest_feats, interest_graph,link2id,domain2id,index2interests)
    model_param = torch.load(model_states)
    interest_model.load_state_dict(model_param)
    interest_model.eval()
    pnet = interest_model.product_model
    inet = interest_model.interest_model
    with torch.no_grad():
        h_p = pnet(False, testset)
        h_i = inet()
        h_p = h_p.cpu().numpy()
        h_i = h_i.cpu().numpy()
    i, j = h_p.shape
    h_p = h_p.reshape(i, 1, j)
    i, j = h_i.shape
    h_i = h_i.reshape(1, i, j)
    out = np.sum(h_p * h_i, axis=-1)
    indices = np.argsort(out, axis=1)[:,::-1] # reverse order
    topk = 20
    for idx, (i, j) in enumerate(test_link_feats):
        print(f'the interest words selected for link {(i,j)} is: {[interest_feats[q][1] for q in indices[idx][:topk]]}')
    print('Done printing')

if __name__ =='__main__':
    test_links_path = argv[1]
    link2id_path = argv[2]
    domain2id_path = argv[3]
    interest_feats_path = argv[4]
    interest_graph_path = argv[5]
    interests2idx_path = argv[6]
    model_state_path = argv[7]
    interests2idx = pickle_load(interests2idx_path)
    index2interests = dict([(j,i) for (i,j) in interests2idx.items()])
    test_links = pickle_load(test_links_path)
    link2id =  pickle_load(link2id_path)
    domain2id = pickle_load(domain2id_path)
    interest_feats = pickle_load(interest_feats_path)
    interest_graph = pickle_load(interest_graph_path)
    _ = interest_model_inference(test_links,link2id,domain2id,interest_feats,interest_graph,index2interests,model_state_path)



