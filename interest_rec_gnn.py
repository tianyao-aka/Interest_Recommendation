# -*- coding:utf-8 -*-
from __future__ import absolute_import, print_function, division
import networkx as nx
import os
import sys
import json
import traceback
from random import choices
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
import pandas as pd
import numpy as np
import pickle
import math
import torch
from nltk.corpus import words
import matplotlib.pyplot as plt
from collections import Counter
import torch.nn.utils as utils
import networkx as nx
from sys import argv



# define gnn model and dataloader
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.optim as optim
from torchnlp.word_to_vector import FastText
import re

fasttext_vec = FastText()
device = torch.device('cpu')

def parse_link_noun_phrase(x):
    # parse the noun phrase in the http link
    try:
        out = []
        x = x.replace('https://','')
        x = x.split('/')
        x = [i for i in x if '.com' not in i and 'collection' not in i and 'product' not in i]
        x = [i.replace('-',' ') if '?' not in i else i.split('?')[0].replace('-',' ') for i in x if '-' in i or (' ' not in i)]   #and i in vectors
        return ''.join(x)
    except:
        return ''


def remove_longtail_interests(df_data):
    # remove longtail interest words.
    # input: data frame for the dataset
    # output: a set of (targets, #occurence_targets)
    candidate_targets = set()
    targets = df_data['interests'].tolist()
    targets = [j for i in targets for j in i]
    target_counter= dict(Counter(targets))
    for t in target_counter:
        if target_counter[t]>1:
            candidate_targets.add((t,target_counter[t]))
    return candidate_targets


def remove_interests_in_df(df_data):
    """
    this method filters the rows with empty interest words
    :param df_data: dataframe of the dataset
    :return: reduced dataframe of the dataset
    """

    candidate_targets = remove_longtail_interests(df_data)
    candidate_targets = set([i[0] for i in candidate_targets])
    df_data['interests'] = df_data['interests'].apply(lambda x:[i for i in x if i in candidate_targets])
    print (f'before df remove interests: {df_data.shape}')
    mask = df_data['interests'].apply(lambda x:len(x)>0)
    df_data = df_data[mask]
    print (f'after df remove interests: {df_data.shape}')
    return df_data


def indexing_links(df_data):
    """
    this method index each link in the dataset
    :param df_data:
    :return: return a dict of link to index
    """
    links = set(df_data['link'].tolist())
    link2index = dict(zip(links,np.arange(len(links))))
    return link2index


def indexing_interests(candidate_interests):
    """
    this method index each interest word
    :param candidate_interests:  a set of (interest_word,num_occurence)
    :return: a dict of interest to index
    """
    temp = [i[0] for i in candidate_interests]
    interest2index = dict(zip(temp, np.arange(len(temp))))
    return interest2index


def build_graph(df_data, product2index, interest2index):
    """
    this method build bipartite graph consisting of two types of nodes:(product_link,interest word)
    :param df_data: the same as before
    :param product2index: a dict of (link,index)
    :param interest2index: a dict of (interest,index)
    :return: adjacency matrix of shape: (num_link,num_interest)
    """
    num_products = len(product2index)
    num_interests = len(interest2index)
    graph = np.zeros((num_products, num_interests))
    for k in df_data.iterrows():
        k = k[1]
        link = product2index[k['link']]
        targets = k['interests']
        targets = [interest2index[i] for i in targets]
        graph[link, targets] = 1
    return graph


# build product graph

def build_product_graph(graph, thres=7):
    """
    Deprecated
    a method that builds product_link graph using the bipartite matrix
    :param graph: a bipartite adjacency matrix
    :param thres: threshold for linking an edge
    :return: link_graph and node list that inside a connected component bigger than 30 nodes
    """
    avaliable_nodes = []
    product_graph = graph.dot(graph.T)
    product_graph[product_graph <= thres] = 0
    product_graph[product_graph > thres] = 1
    indices = np.where(product_graph == 1)
    N = product_graph.shape[0]
    G = nx.Graph()
    for i in range(len(indices[0])):
        G.add_edge(indices[0][i], indices[1][i])
    for i in range(N):
        G.add_edge(i, i)

    compo = nx.connected_components(G)
    for c in compo:
        if len(c) > 30:
            print(len(c))
            for i in c:
                avaliable_nodes.append(i)
    return G, avaliable_nodes


def build_interest_graph(graph, thres=30):
    """
    similar to build_product_graph
    :param graph:
    :param thres:
    :return:
    """
    avaliable_nodes = []
    interest_graph = graph.T.dot(graph)
    interest_graph[interest_graph <= thres] = 0
    interest_graph[interest_graph > thres] = 1
    indices = np.where(interest_graph == 1)
    N = interest_graph.shape[1]
    G = nx.Graph()
    for i in range(len(indices[0])):
        G.add_edge(indices[0][i], indices[1][i])
    for i in range(N):
        G.add_edge(i, i)

    compo = nx.connected_components(G)
    for c in compo:
        if len(c) > 30:
            print(len(c))
            for i in c:
                avaliable_nodes.append(i)
    return G, avaliable_nodes


def process_text(text):
    """
    preprocess ad text
    """
    text = text.replace("【", "(")
    text = text.replace("】", ")")
    text = text.replace("[", "(")
    text = text.replace("]", ")")
    text = text.replace("（", "(")
    text = text.replace("）", ")")
    return text.replace('(', ' ').replace(')', ' ').replace('-', ' ')


def get_single_sent_emb(vectors, sent, device=torch.device('cpu')):
    """
    this method return the representation of a single sentence, or a phrase.
    :param vectors: fasttext word vector
    :param sent: input sentence
    :param device:
    :return: mean vector of a given sentence
    """
    # print ("%%%%%%%%%%%:",sent)
    try:
        emb = [vectors[[x]] for x in sent.split()]
        emb = torch.mean(torch.cat(emb, dim=0), dim=0).view(1, -1)
        return emb.to(device)
    except:
        return vectors[['000']].to(device)


# extract website domain name
def extract_domain_name(x):
    """
    this method extract domain name for a given product link
    :param x:  a product link
    :return: a string of domain name
    """
    if 'gnoce' in x.lower():
        return 'gnoce'
    regex_string = r".*//[www.]*(.*).com/"
    t = re.findall(regex_string, x)
    if len(t) > 0:
        return t[0]
    return 'empty'


# indexing domain name
def indexing_domain_name(x):
    """
    this method takes a list of domain names and adding frequent domains into a set
    :param x: list of domains
    :return: domain set
    """
    domain_set = set()
    q = dict(Counter(x).most_common(5000))
    for i in q:
        if i == 'empty':
            continue
        if q[i] < 30:
            break
        domain_set.add(i)
    return domain_set


# get all interests by item link
def get_all_positive_interests_by_link(df_data):
    """
    a method to build dict of key:link, value:interests
    :param df_data: given dataframe dataset
    :return: dict of (key:link, value:interests)
    """
    def get_all_interests_by_link(x):
        return [j for i in x for j in i]

    q = df_data[['link', 'interests']].groupby(['link'], as_index=False).agg({'interests': get_all_interests_by_link})
    link2all_interests = dict(zip(q['link'].tolist(), q['interests'].tolist()))
    return link2all_interests


# process data to get dataset
def process_dataset(df_data, link2all_interests):
    """
    a method takes raw data and process it into formatted dataset
    :param df_data: raw dataset
    :param link2all_interests: dict of (link,interest_words)
    :return: processed dataset
    """
    df_dataset = []
    r = r"[0-9]*"

    def process_dataset_(x):
        link_nouns = parse_link_noun_phrase(x['link'])
        all_interests = link2all_interests[x['link']]
        title = x['title']
        product_desc = title if title != '000' else link_nouns
        product_desc = re.sub(r, '', product_desc)
        product_desc = product_desc.lower()
        link_id = link2index[x['link']]
        spend = x['spend']
        df_dataset.append(
            [link_id, x['link'], product_desc, x['domain_name'], x['interests'], all_interests, x['roas'], spend])

    _ = df_data.apply(process_dataset_, axis=1)
    df_dataset = pd.DataFrame(df_dataset,
                              columns=['link_id', 'link', 'description', 'domain_name', 'interests', 'all_interests',
                                       'roas', 'spend'])
    return df_dataset


# get most common links
def get_most_common_link(df):
    """
    a method to put frequent links to a set
    :param df: input dataset
    :return: a set holding frequent links
    """
    link_set = set()
    q = dict(Counter(df['link_id'].tolist()).most_common(5000))
    for i in q:
        if q[i] < 11:
            break
        else:
            link_set.add(i)
    return link_set


# clean descriptions
def clean_description(x):
    """
    remove punctuations of a given string
    :param x: a string of text
    """
    re_str = r"[!,?=&.\-\:@\|]+"
    x = re.sub(re_str, ' ', x)
    return x


# interests occurs more than 100 times
def get_most_common_interest(candidate_interests):
    # get frequent occuring interest words
    return set([interests2index[i[0]] for i in candidate_interests if i[1] > 30])


def commonlink2index_(x, commonlink2index):
    """
    link to embedding id
    :param x: input product_link
    :param commonlink2index: dict of frequent link to index
    :return:
    """
    N = len(commonlink2index)
    if x in commonlink2index:
        return commonlink2index[x]
    else:
        return N # if link not in the dict, return N meaning "other"


def commondomain2index_(x, commondomain2index):
    # similar to commonlink2index_
    N = len(commondomain2index)
    if x in commondomain2index:
        return commondomain2index[x]
    elif x != 'empty':
        return N  # if not empty, return N
    return N + 1 # if empty, return N+1



def commoninterest2index_(x, commoninterest2index):
    # similar to commonlink2index_
    N = len(commoninterest2index)
    if x in commoninterest2index:
        return commoninterest2index[x]
    else:
        return N


# build link features: $link_id, $description, $domain_name
def build_item_feature_matrix(df_dataset):
    """
    build product link features: ($link_id, $description, $domain_name)
    :param df_dataset: input dataset
    :return: list of link features
    """
    feats = []
    links = set()
    for _, i in df_dataset.iterrows():
        if i['link_id'] in links: continue
        feats.append([i['link_id'], i['description'], i['domain_name']])
        links.add(i['link_id'])
    feats = sorted(feats, key=lambda x: x[0])
    return feats


# build interest feature matrix
def build_interest_feature_matrix(index2interests):
    """
    build interest words feature matrix
    :param index2interests: dict of (index,interest_words)
    :return: list of (index,interest_words)
    """
    out = sorted(list(index2interests.items()), key=lambda x: x[0])
    return out


# add column for interest index
def add_interest_index(ds):
    """
    add interest word id as a new column
    :param ds: row data series in the apply function
    :return:
    """
    interests = ds['interests']
    all_interests = ds['all_interests']
    interests_id = set([interests2index[i] for i in interests])
    all_interests_id = set([interests2index[i] for i in all_interests])
    ds['interest_id'] = interests_id
    ds['all_interest_id'] = all_interests_id
    return ds


# sample a minibatch for BPR loss
def sample_minibatch(df_dataset,candidate_interest_id,candidate_interest_weights):
    """
    invoke the function sample_one_record to sample a minibatch in the dataset
    :param df_dataset: input dataset
    :param candidate_interest_id:
    :param candidate_interest_weights:
    :return:
    """
    N = df_dataset.shape[0]
    spend = np.sqrt(np.asarray(df_dataset['spend'].tolist()))
    spend = spend/np.sum(spend)
    minibatch = np.random.choice(np.arange(N),size = 512,replace = False,p =spend)
    triplets = []
    for i in minibatch:
        tri = sample_one_record(df_dataset,i,candidate_interest_id,candidate_interest_weights)
        triplets.append(tri)
    return triplets


def sample_one_record(df_dataset,index,candidate_interest_id,candidate_interest_weights):
    """
    sample a single record for model training
    :param df_dataset: input dataset
    :param index: row index for df_dataset
    :param candidate_interest_id: all the candidate interest ids
    :param candidate_interest_weights: all the candidate interest weights
    :return: a 4-tuple:(query_link_id,positive_sample_link_id,30 negtive_link_ids, roas of the advertising record)
    """
    q = df_dataset.iloc[index]
    query = q['link_id']
    roas = q['roas']
    pos = np.random.choice(list(q['interest_id']))
    neg =  np.random.choice(candidate_interest_id, p =candidate_interest_weights,size = 30,replace = False)
    neg = list(set(neg) - q['all_interest_id'])
    if len(neg)==0:
        print ('neg_len = 0 after set removal')
    neg_ = np.random.choice(neg)
    return (query,pos,neg_,roas)

import math

link_recall_cache = {}

def build_interest_roas_dict(df):
    """
    build a dict of {interest:roas,...} using inverted indexing
    :param df: df dataset
    :return: a dict of {interest:roas,...}
    """
    interest2roas = {}
    all_interests = set()
    interest_ = df['interest_id'].tolist()
    for i in interest_:
        for q in i:
            all_interests.add(q)
    for k in all_interests:
        mask = df['interest_id'].apply(lambda x: k in x)
        roas = np.asarray(df[mask]['roas'].tolist())
        spend = np.asarray(df[mask]['spend'].tolist())
        wroas = np.sum(roas * spend) / np.sum(spend)
        interest2roas[k] = wroas
    return interest2roas


def add_avg_roas_col(ds):
    """
    compute average roas for a product link
    :param ds: row data series of input dataset
    :return: new data seris with column name: avg_roas
    """
    tmp = ds['interest_id']
    roas = np.asarray([interest_roas_dict[i] for i in tmp])
    ds['avg_roas'] = np.mean(roas)
    return ds


def sigmoid(x):
    return 1. / (1 + math.exp(-x))


def recall_interests(df, index, pfeats, ifeats, index2interest):
    """
    recall the interest words for a given link id
    :param df: input dataset
    :param index: roe indice for the dataset
    :param pfeats: representation from product_net
    :param ifeats: representation from interest_net
    :param index2interest:
    :return: a triplet of (interest words of the given record, selected interests from the model, all interests for the given product)
    """
    q = df.iloc[index]
    selected_interests = []
    link_id = q['link_id']
    title = q['description']
    link_addr = q['link']
    interests = list(set(q['interests']))
    all_interests = list(set(q['all_interests']))
    if link_id not in link_recall_cache:
        a = pfeats[[link_id]]
        b = ifeats
        dotsum = np.sum(a * b, axis=1)
        indices = np.argsort(dotsum)[::-1]
        for i in indices:
            if sigmoid(dotsum[i]) > 0.75 or len(selected_interests) < 20:
                selected_interests.append(index2interest[i])

        link_recall_cache[link_id] = selected_interests
        return interests, selected_interests, all_interests
    else:
        selected_interests = link_recall_cache[link_id]
        return interests, selected_interests, all_interests
    # return link_id,title,link_addr,selected_interests,set(q['all_interests'])


def calc_samplewise_eval_metric(interests, selected_interests, all_interests):
    interest_vec = np.asarray([get_single_sent_emb(fasttext_vec, i).numpy() for i in interests])
    interest_vec = interest_vec.squeeze(axis=1)
    all_interests_vec = np.asarray([get_single_sent_emb(fasttext_vec, i).numpy() for i in all_interests])
    all_interests_vec = all_interests_vec.squeeze(axis=1)
    model_selected_interests_vec = np.asarray(
        [get_single_sent_emb(fasttext_vec, i).numpy() for i in selected_interests])
    model_selected_interests_vec = model_selected_interests_vec.squeeze(axis=1)
    sim_score_vec = cosine_similarity(interest_vec, model_selected_interests_vec)
    semantic_related_score = np.mean(np.max(sim_score_vec, axis=1))
    diverse_score = 0
    q = np.min(sim_score_vec, axis=0)
    q = q < 0.35
    if np.sum(q) > 0:
        diverse_interest_vec = model_selected_interests_vec[q]
        diverse_score_vec = cosine_similarity(diverse_interest_vec, all_interests_vec)
        q = np.max(diverse_score_vec, axis=1)
        diverse_score = np.sum(q >= 0.7) / model_selected_interests_vec.shape[0]
    return semantic_related_score, diverse_score


def calc_avg_roas_from_model_output(selected_interests):
    """
    given selected output from model, calculate its average roas
    :param selected_interests: interests from model output
    :return: average roas for a set of selected interests
    """
    selected_interest_id = [interests2index[i] for i in selected_interests]
    roas = np.asarray([interest_roas_dict[i] for i in selected_interest_id if i in interest_roas_dict])
    if len(roas) == 0:
        return 0.
    return np.mean(roas)


def pickle_dump(to_pickle_file,file_name):
    with open(file_name,'wb') as f:
        pickle.dump(to_pickle_file,f)
    return

# define gnn model and dataloader

class Text_Emb_Model(nn.Module):
    def __init__(self):
        super(Text_Emb_Model, self).__init__()
        self.linear1 = nn.Linear(300, 168)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(168, 80)

    def forward(self, text):
        return self.linear2(self.relu(self.linear1(text)))




class Product_NN(nn.Module):
    def __init__(self, link_feats, text_emb_net,link2id=None,domain2id = None):
        super(Product_NN, self).__init__()
        self.feats = link_feats
        self.text_feats = torch.cat([get_single_sent_emb(fasttext_vec, i[1]) for i in self.feats], dim=0)
        if link2id is None and domain2id is None:
            N_link = len(commonlink2index)
            N_domain = len(commondomain2index)
        else:
            N_link = len(link2id)
            N_domain = len(domain2id)
        self.text_emb_model = text_emb_net

        self.link_emb = nn.Embedding(N_link + 1, 50)
        self.domain_emb = nn.Embedding(N_domain + 2, 30, padding_idx=N_domain + 1)
        self.linear1 = nn.Linear(160, 80)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(80, 18)

    def transform_feats(self):
        link_emb_id = self.link_emb(torch.tensor([i[0] for i in self.feats]).long())
        domain_emb_id = self.domain_emb(torch.tensor([i[2] for i in self.feats]).long())
        text_feats = self.text_emb_model(self.text_feats)
        return torch.cat((link_emb_id, text_feats, domain_emb_id), dim=1).to(device)

    def forward(self,train_stage = True,link_feats = None):
        """

        :param train_stage: whether it's in training stage or inference stage
        :param link_feats: [(link_id,text,domain_id),.....]
        :return: embeddings for each product link
        """
        if train_stage:
            x = self.transform_feats()
            h = self.linear2(self.relu(self.linear1(x)))
            return h
        else:
            assert link_feats is not None
            text_feats = torch.cat([get_single_sent_emb(fasttext_vec, i[1]) for i in link_feats], dim=0)
            text_feats = self.text_emb_model(text_feats)
            link_emb_id = self.link_emb(torch.tensor([i[0] for i in link_feats]).long())
            domain_emb_id = self.domain_emb(torch.tensor([i[2] for i in link_feats]).long())
            h = torch.cat((link_emb_id, text_feats, domain_emb_id), dim=1).to(device)
            h = self.linear2(self.relu(self.linear1(h)))
            return h


class Interest_GNN(nn.Module):
    def __init__(self, interest_feats, interest_graph, text_emb_net,interest2index=None):
        super(Interest_GNN, self).__init__()
        try:
            N = len(commoninterest2index)
        except:
            N = len(interest2index)
        self.feats = interest_feats
        self.graph = interest_graph
        self.text_emb_model = text_emb_net
        self.edge_index = torch.tensor(list(interest_graph.edges)).permute(1, 0).contiguous().long()
        self.text_feats = torch.cat([get_single_sent_emb(fasttext_vec, i[1]) for i in self.feats], dim=0)

        self.text_emb_model = text_emb_net
        self.interest_emb = nn.Embedding(N + 1, 50)
        self.gcn1 = GCNConv(130, 60)
        self.dropout = nn.Dropout()
        self.relu2 = nn.ReLU()
        self.gcn2 = GCNConv(60, 30)
        self.relu3 = nn.ReLU()
        self.linear3 = nn.Linear(30, 18)

    def transform_feats(self):
        interest_emb_id = self.interest_emb(torch.tensor([i[0] for i in self.feats]).long())
        text_feats = self.text_emb_model(self.text_feats)
        return torch.cat((interest_emb_id, text_feats), dim=1).to(device)

    def forward(self):
        x = self.transform_feats()
        h = self.relu2(self.gcn2(self.dropout(self.relu2(self.gcn1(x, self.edge_index))), self.edge_index))
        h = self.linear3(h)
        return h


class Interest_Rec_Model(nn.Module):
    def __init__(self, link_feats, interest_feats, interest_graph,link2id=None,domain2id=None,interest2id = None):
        super(Interest_Rec_Model, self).__init__()
        text_emb_model = Text_Emb_Model()
        self.product_model = Product_NN(link_feats, text_emb_model,link2id,domain2id)
        self.interest_model = Interest_GNN(interest_feats, interest_graph, text_emb_model,interest2id)

    def forward(self):
        product_feats = self.product_model()
        interest_feats = self.interest_model()
        return product_feats, interest_feats


## nn baseline model

class Interest_NN(nn.Module):
    def __init__(self, interest_feats, text_emb_net):
        super(Interest_NN, self).__init__()
        N = len(commoninterest2index)
        self.feats = interest_feats
        self.text_emb_model = text_emb_net
        self.text_feats = torch.cat([get_single_sent_emb(fasttext_vec, i[1]) for i in self.feats], dim=0)
        self.text_emb_model = text_emb_net
        self.interest_emb = nn.Embedding(N + 1, 50)
        self.linear1 = nn.Linear(130, 70)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(70, 30)

    def transform_feats(self):
        interest_emb_id = self.interest_emb(torch.tensor([i[0] for i in self.feats]).long())
        text_feats = self.text_emb_model(self.text_feats)
        return torch.cat((interest_emb_id, text_feats), dim=1).to(device)

    def forward(self):
        x = self.transform_feats()
        h = self.linear2(self.relu(self.linear1(x)))
        return h


class Interest_Rec_NN_Model(nn.Module):
    def __init__(self, link_feats, interest_feats):
        super(Interest_Rec_NN_Model, self).__init__()
        text_emb_model = Text_Emb_Model()
        self.product_model = Product_NN(link_feats, text_emb_model)
        self.interest_model = Interest_NN(interest_feats, text_emb_model)

    def forward(self):
        product_feats = self.product_model()
        interest_feats = self.interest_model()
        return product_feats, interest_feats


if __name__ == '__main__':
    data_path = argv[1]
    link_path = argv[2]
    # load link2title file
    with open(data_path, 'rb') as f:   # load the raw df dataset
        df_data = pickle.load(f)
    with open(link_path, 'rb') as f:   # load the crawled dictionary of (key:link,value:title)
        link2title = pickle.load(f)
    df_data['title'] = df_data['link'].apply(lambda x: link2title[x] if x in link2title else '000')
    df_data['domain_name'] = df_data['link'].apply(extract_domain_name) # extarct domain name for each link
    domain_set = indexing_domain_name(df_data['domain_name'].tolist())

    df_data = remove_interests_in_df(df_data)  # remove interest words with empty string
    link2index = indexing_links(df_data)
    candidate_interests = remove_longtail_interests(df_data) # obtain frequent occuring interests
    interests2index = indexing_interests(candidate_interests)
    index2link = dict([(j, i) for i, j in link2index.items()])
    index2interests = dict([(j, i) for i, j in interests2index.items()])
    graph = build_graph(df_data, link2index, interests2index)   # build bipartite graph
    product_graph, product_nodes = build_product_graph(graph)   # product co-occurence graph, not used
    interest_graph, interest_nodes = build_interest_graph(graph, 35) # build interest graph and its nodes in the graph
    link2all_interests = get_all_positive_interests_by_link(df_data) # get all interests for each product link
    linkid2all_interests = dict([(link2index[i], j) for i, j in link2all_interests.items()])
    df_dataset = process_dataset(df_data, link2all_interests)   # process raw dataset to parse noun phrase from links,and add positive interests for each link
    link_set = get_most_common_link(df_dataset)  # get frequent links
    # filter dataset with invalid descriptions
    mask = df_dataset['description'].apply(lambda x: '.' not in x and 'http' not in x)
    df_dataset['description'] = df_dataset['description'].apply(clean_description)
    interest_set = get_most_common_interest(candidate_interests)

    # initialize commonlink2idex and commondomain2index, commoninterest2index
    commonlink2index = dict(zip(link_set, np.arange(len(link_set))))
    commondomain2index = dict(zip(domain_set, np.arange(len(domain_set))))
    commoninterest2index = dict(zip(interest_set, np.arange(len(interest_set))))
    pickle_dump(commonlink2index,'link2id.pkl') # dump file for inference stage
    pickle_dump(commondomain2index, 'domain2id.pkl') # dump file for inference stage
    pickle_dump(commoninterest2index,'interest2id.pkl') # dump file for inference stage
    # build link feats
    feats = build_item_feature_matrix(df_dataset)
    link_feats = [(commonlink2index_(i, commonlink2index), j, commondomain2index_(k, commondomain2index)) for i, j, k in
                  feats]
    link_feats = [(i, j, k) if mask[idx] else (i, '000', k) for idx, (i, j, k) in enumerate(link_feats)]
    # build interest feats
    feats = build_interest_feature_matrix(index2interests)
    interest_feats = [(commoninterest2index_(i, commoninterest2index), j) for i, j in feats]
    pickle_dump(interest_feats,'interest_feats.pkl')
    pickle_dump(interest_graph, 'interest_graph.pkl')
    # remove useless samples
    df_dataset = df_dataset[mask]

    # if link is rare and domain is rare and can't parse noun phrase from link address, remove it.
    N_link = len(commonlink2index)
    N_domain = len(commondomain2index)
    remove_links = []
    for i in range(len(link_feats)):
        if link_feats[i][0] == N_link and (link_feats[i][2] == N_domain or link_feats[i][2] == N_domain + 1) and 'http' in \
                link_feats[i][1]:
            remove_links.append(i)
    # remove nodes from graph
    for n in remove_links:
        product_graph.remove_node(n)

    # remove link_id from df_dataset
    remove_links = set(remove_links)
    remove_mask = df_dataset['link_id'].apply(lambda x: x not in remove_links)
    df_dataset = df_dataset[remove_mask]

    # add index to interest words
    print('add index')
    df_dataset = df_dataset.apply(lambda x: add_interest_index(x), axis=1)

    # candidate interests word id and weights
    candidate_interest_id = [interests2index[i[0]] for i in candidate_interests]
    candidate_interest_weights = np.asarray([math.sqrt(i[1]) for i in candidate_interests])
    candidate_interest_weights = candidate_interest_weights / np.sum(candidate_interest_weights)

    # split train and test dataset
    import random

    product_nodes = list(product_graph.nodes)
    random.Random(6615).shuffle(product_nodes)
    N = len(product_nodes)
    train_products = set(product_nodes[:int(0.75 * N)])
    test_products = set(product_nodes[int(0.75 * N):])
    print('entire dataset:', df_dataset.shape)
    train_df_dataset = df_dataset[df_dataset['link_id'].apply(lambda x: x in train_products)]
    test_df_dataset = df_dataset[df_dataset['link_id'].apply(lambda x: x in test_products)]
    print('train df shape:', train_df_dataset.shape)
    print('test df shape:', test_df_dataset.shape)

    # build interest to roas dict
    interest_roas_dict = build_interest_roas_dict(df_dataset)
    # add a new column for average roas in test dataset
    test_df_dataset = test_df_dataset.apply(add_avg_roas_col, axis=1)
    # train model
    interest_model = Interest_Rec_Model(link_feats, interest_feats,interest_graph)
    optimizer = optim.Adam(interest_model.parameters(), lr=5e-3, weight_decay=2e-6)
    nn_loss_tracker = []
    for e in range(15000):
        minibatch = sample_minibatch(train_df_dataset, candidate_interest_id, candidate_interest_weights)
        query = torch.tensor([i[0] for i in minibatch]).long()
        pos = torch.tensor([i[1] for i in minibatch]).long()
        neg = torch.tensor([i[2] for i in minibatch]).long()
        roas = torch.tensor([1. if i[3] < 1 else i[3] for i in minibatch]).float()
        product_out_feat, interest_out_feat = interest_model() # forward to get product embedding and interest embedding
        product_ = product_out_feat[query] # query embedding
        interest_pos = interest_out_feat[pos] # positive interest embedding
        interest_neg = interest_out_feat[neg] # negative interest embedding
        pos_multiply = torch.sum(product_ * interest_pos, dim=1)
        neg_multiply = torch.sum(product_ * interest_neg, dim=1)
        loss_val = -1. * torch.log(torch.sigmoid(pos_multiply - neg_multiply))  # BPR loss
        loss_val = torch.mean(loss_val * roas) # weighted by ROAS
        if math.isnan(loss_val.item()) or math.isinf(loss_val.item()):
            continue
        if e < 10:
            print('mean loss is:', loss_val)
        nn_loss_tracker.append(loss_val.item())
        if e < 500 and e % 50 == 0:
            print(f'current step {e}')

        if e >= 500 and e % 100 == 0:
            mean_loss = sum(nn_loss_tracker[-400:]) / 400.
            print(f"mean loss for step {e - 400} to {e} is:", mean_loss)
        loss_val.backward()
        utils.clip_grad_value_(interest_model.parameters(), 5)
        optimizer.step()
        optimizer.zero_grad()
        if e % 1000 == 0 and e > 0:
            torch.save(interest_model.state_dict(), 'interest_rec_model_checkpoints.pt')
    print ('finished')
