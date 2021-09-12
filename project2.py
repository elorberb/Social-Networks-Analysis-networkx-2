from builtins import print
import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp
import tqdm as tm
import pickle as pk
import community as community_louvain
import matplotlib.pyplot as plt
from networkx.algorithms import community
from networkx.algorithms import approximation
from collections import defaultdict
import json
from datetime import datetime
import datetime as dt
from itertools import product


def get_name():  # name function
    name = "itay lorebrboym"
    print(name)
    return name

def get_id():  # id function
    id = "314977596"
    print(id)
    return id

# ------------------------ 1 ---------------------- Community detection wrapper

# -- i --

def modularity_overlapping(G, communities, weight='weight'):
    if not isinstance(communities, list):
        communities = list(communities)

    multigraph = G.is_multigraph()
    directed = G.is_directed()
    m = G.size(weight=weight)
    if directed:
        out_degree = dict(G.out_degree(weight=weight))
        in_degree = dict(G.in_degree(weight=weight))
        norm = 1 / m
    else:
        out_degree = dict(G.degree(weight=weight))
        in_degree = out_degree
        norm = 1 / (2 * m)

    def val(u, v):
        try:
            if multigraph:
                w = sum(d.get(weight, 1) for k, d in G[u][v].items())
            else:
                w = G[u][v].get(weight, 1)
        except KeyError:
            w = 0
        # Double count self-loops if the graph is undirected.
        if u == v and not directed:
            w *= 2
        return w - in_degree[u] * out_degree[v] * norm

    Q = sum(val(u, v) for c in communities for u, v in product(c, repeat=2))
    return Q * norm

def community_detector(algorithm_name, network, most_valualble_edge=None):
    if (algorithm_name == "girvin_newman"):
        g_n_communities = list(community.girvan_newman(network,most_valualble_edge))
        modularity = 0
        partition = []
        for communities in g_n_communities:
            current_partition = tuple(sorted(c) for c in communities)
            current_modularity = community.modularity(network, current_partition)
            if current_modularity > modularity:
                modularity = current_modularity  # modularity
                partition = list(current_partition)  # partition
        num_partitions = len(partition)  # number of communities

    elif (algorithm_name == "louvain"):
        communities = community_louvain.best_partition(network)
        new_dict = defaultdict(list)
        for key, value in communities.items():
            new_dict[value].append(key)
        partition = [c for c in new_dict.values()]  # creatiing partition
        num_partitions = len(partition)  # number of comuunities
        modularity = community_louvain.modularity(communities, network)  # modularity

    elif (algorithm_name == "clique_percolation"):
        largest_clique = nx.algorithms.approximation.clique.large_clique_size(network)  # search for largest clique
        partition = list(list(c) for c in list(community.k_clique_communities(network, 3)))
        modularity = modularity_overlapping(network, partition)
        for k in range(4, largest_clique + 1, 1):  # checks for best modularity of every possible k community
            current_partition = list(list(c) for c in list(community.k_clique_communities(network, k)))
            current_modularity = modularity_overlapping(network, current_partition)
            if (current_modularity > modularity):
                modularity = current_modularity  # best modularity
                partition = current_partition  # best partition based on modularity
        num_partitions = len(partition)  # number of communities

    communities_partitions = {'num_partitions': num_partitions, 'modularity': modularity, 'partition': partition}

    return communities_partitions

def print_community_info(communities_partitions):
    print("Number of partitions: ", communities_partitions['num_partitions'])
    print("Modularity: ", communities_partitions['modularity'])
    print("Partitions:")
    for i in range(len(communities_partitions['partition'])):
        print("Community", i, "-",communities_partitions['partition'][i])
    return

# -- ii --

def edge_selector_optimizer(network):
    if not nx.is_connected(network):
        betweenness = nx.edge_betweenness_centrality(network)
        return max(betweenness, key=betweenness.get)

    soc_nodes = sorted(nx.second_order_centrality(network).items(), key=lambda x: x[1])
    cb_nodes = sorted(nx.communicability_betweenness_centrality(network).items(), key=lambda x: x[1], reverse=True)

    for socn in soc_nodes:
        bsocn = socn[0]
        for cbn in cb_nodes:
            name = cbn[0]
            if name in network[bsocn]:
                return (name, bsocn)

# ------------------------ 2 ---------------------- Community detection - Twitter data

# -- i --

def get_central_political_players(path):
    political_df = pd.read_csv(path)
    plotical_dict = political_df.set_index('id').to_dict()
    return plotical_dict["name"]

def build_retweet_dict(all_tweets):
    ans = {}
    for t in all_tweets:
        tweet_data = json.loads(t)
        tweet_user = tweet_data["user"]['id']
        try:
            retweet_user = tweet_data["retweeted_status"]["user"]['id']
            edge = (retweet_user, tweet_user)
            if edge in ans:
                ans[edge] += 1
            else:
                ans[edge] = 1
        except:
            pass
    ans = dict(sorted(ans.items(), key=lambda item: item[1], reverse=True))
    return ans

def construct_heb_edges(files_path, start_date='2019-03-15', end_date='2019-04-15', non_parliamentarians_nodes = 0):
    delta = dt.timedelta(days=1)
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    all_tweets = []

    while start_date <= end_date: #adding all retweets between dates to the list
        start_date = datetime.strftime(start_date, '%Y-%m-%d')
        end_date = datetime.strftime(end_date, '%Y-%m-%d')
        file_full_path = files_path + "\Hebrew_tweets.json." + start_date + ".txt"
        current_tweets_file = open(file_full_path, 'r')
        current_tweets_str = current_tweets_file.readlines()
        current_tweets_str = [t.strip() for t in current_tweets_str]
        all_tweets.extend(current_tweets_str)
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        start_date += delta
    retweets_dict = build_retweet_dict(all_tweets)

    central_political_players_path = files_path + "\central_political_players.csv"
    central_political_players = get_central_political_players(central_political_players_path)
    central_political_players_id = list(central_political_players.keys()) #building list of the poltical players id

    if non_parliamentarians_nodes < len(retweets_dict):
        extra_non_parliamentarians_nodes = add_non_parliamentarians_nodes(non_parliamentarians_nodes,retweets_dict,central_political_players_id)
        retweets_dict = remove_non_central_players(retweets_dict,central_political_players_id)
    ans = connect_dicts(retweets_dict,extra_non_parliamentarians_nodes) #connecting the dictionaries
    names_dict = id_to_names(central_political_players,ans)

    return names_dict

def connect_dicts(dict1,dict2):
    copy1 = dict(dict1)
    ans = {}
    ans.update(copy1)
    if len(dict2) != 0:
        for key,value in dict2.items():
            if key in ans.keys():
                ans[key] += value
            else:
                ans[key] = value
    return ans

def remove_non_central_players(ans,central_political_players_id):
    for edge in [edge for edge in ans
                 if (edge[0] not in central_political_players_id) or (edge[1] not in central_political_players_id)]: del ans[edge]
    return ans

def add_non_parliamentarians_nodes(non_parliamentarians_nodes,ans,central_political_players_id):
    extra_non_parliamentarians_nodes = {}
    counter = non_parliamentarians_nodes
    poli_to_non_poli_retweet = {}
    non_poli_to_poli_retweet = {}
    non_poli_to_non_poli_retweet = {}
    if non_parliamentarians_nodes == 0:
        return extra_non_parliamentarians_nodes
    for edge in ans: # divide retweets by the user
        if (edge[0] in central_political_players_id and edge[1] not in central_political_players_id):
            poli_to_non_poli_retweet[edge] = ans.get(edge)
        if (edge[1] in central_political_players_id and edge[0] not in central_political_players_id):
            non_poli_to_poli_retweet[edge] = ans.get(edge)
        if (edge[0] not in central_political_players_id and edge[1] not in central_political_players_id):
            non_poli_to_non_poli_retweet[edge] = ans.get(edge)
    if non_parliamentarians_nodes < len(poli_to_non_poli_retweet): #only adding political nodes that retweeted non political
        for edge in poli_to_non_poli_retweet:
            if (edge[0] in central_political_players_id):
                extra_non_parliamentarians_nodes[edge] = ans.get(edge)
                counter = counter - 1
            if counter == 0:
                break
    else: # adding mixed retweets
        for edge in poli_to_non_poli_retweet:#first political nodes to non political nodes
            extra_non_parliamentarians_nodes[edge] = ans.get(edge)
        counter =non_parliamentarians_nodes - len(poli_to_non_poli_retweet)
        for edge in non_poli_to_poli_retweet: #second non political that retweeted political nodes
            extra_non_parliamentarians_nodes[edge] = ans.get(edge)
            counter = counter - 1
            if counter == 0:
                break
        for edge in non_poli_to_non_poli_retweet: #lastly not political taht retweeted non political
            extra_non_parliamentarians_nodes[edge] = ans.get(edge)

    return extra_non_parliamentarians_nodes

def construct_heb_network(edge_dictionary):
    network = nx.DiGraph()
    nodes = []
    edges = []
    weights = []
    for edge in edge_dictionary: #creating arguments of the network from the dict
        nodes.append(edge[0])
        nodes.append(edge[1])
        edges.append(edge)
        weight = edge_dictionary.get(edge)
        weights.append((edge[0], edge[1], weight))

    network.add_nodes_from(nodes)
    network.add_edges_from(edges)
    network.add_weighted_edges_from(weights)

    return network

def id_to_names(political_players_dict, id_dict):
    names_dict = {}
    for edge in id_dict:
        name0 = political_players_dict.get(edge[0])
        name1 = political_players_dict.get(edge[1])
        weight = id_dict.get(edge)
        if name0 is None and name1 is None:
            names_dict[(str(edge[0]), str(edge[1]))] = weight
        elif name0 is None:
            names_dict[(str(edge[0]), name1)] = weight
        elif name1 is None:
            names_dict[(name0, str(edge[1]))] = weight
        else:
            names_dict[(name0, name1)] = weight

    return names_dict

