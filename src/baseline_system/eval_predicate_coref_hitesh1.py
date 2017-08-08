"""
Do proposition coreference
Methodology: We cluster the mentions based on lexical similarity metrics and synset overlap.

Scoring function for mention pair: Two mentions are similar if any word pairs (non-stop words) are similarly spelt / have synset overlap
Coref score achieved by best system: [0.61, 0.73, 0.56, 0.63]

Author: Compiled from Shyam Upadhyay and Rachel Wities by Hitesh Golchha

"""

import sys
sys.path.append('../common')
sys.path.append('../agreement')

import numpy as np

from okr import *
from entity_coref import *
from clustering_common import cluster_mentions
from parsers.spacy_wrapper import spacy_wrapper
from eval_entity_coref import is_stop, similar_words, same_synset,fuzzy_fit, partial_match
from eval_predicate_coref import get_distance_to_root, get_mention_head



import spacy
from munkres import *
from fuzzywuzzy import fuzz
from spacy.en import English
from num2words import num2words
from nltk.corpus import wordnet as wn


# Don't use spacy tokenizer, because we originally used NLTK to tokenize the files and they are already tokenized
nlp = spacy.load('en')


def evaluate_predicate_coref(test_graphs):
    """
    Receives the OKR test graphs and evaluates them for predicate coreference
    :param test_graphs: the OKR test graphs
    :return: the coreference scores: MUC, B-CUBED, CEAF and MELA (CoNLL F1).
    """
    parser = spacy_wrapper()

    scores = []

    for graph in test_graphs:

        # Cluster the mentions
        prop_mentions = []
        for prop in graph.propositions.values():
            for mention in prop.mentions.values():

                if mention.indices == [-1]:
                    continue

                head_lemma, head_pos = get_mention_head(mention, parser, graph)
                prop_mentions.append((mention, head_lemma, head_pos))

        clusters = cluster_mentions(prop_mentions, score)
        clusters = [set([item[0] for item in cluster]) for cluster in clusters]

        # Evaluate
        curr_scores, _ = eval_clusters(clusters, graph)
        scores.append(curr_scores)

    scores = np.mean(scores, axis=0).tolist()

    return scores


def score(prop, cluster):
    """
    Receives a proposition mention (mention, head_lemma, head_pos)
    and a cluster of proposition mentions, and returns a numeric value denoting the
    similarity between the mention to the cluster (% of similar mentions in the cluster) 
    For similarity we compare any two non-stop words of the two mentions for same_synset, partial WordNet match, fuzzy string similarity.

    :param prop: the mention
    :param cluster: the cluster
    :return: a numeric value denoting the similarity between the mention to the cluster
    """
    
    # if you want to compare the dependency heads only for similarity, use this:
    # return len([other for other in cluster if similar_words(other[1],prop[1])]) / (1.0 * len(cluster))


    return len([other for other in cluster if some_word_match(other[0].terms,prop[0].terms)]) / (1.0 * len(cluster))



def some_word_match(string1, string2):
    """
    Receives two strings (maybe multi-word) and finds if any word of string1 (non-stopwords) is similar to any word of string2
    :param string1: the first string
    :param string2: the second string
    :return: boolean value suggesting whether they are similar
    """

    # List of words in the two strings
    words_list1 = string1.split(' ')
    words_list2 = string2.split(' ')
    intersection = []
    for a in words_list1:
        for b in words_list2:
            if(similar_words(a,b)):
                intersection.append(a)
                break

    # If the two strings are not single words, remove Stop Words from the intersection
    if(len(words_list1)!=1 and len(words_list2)!=1):
        intersection = [a for a in intersection if a not in STOP_WORDS]
    return len(intersection)>0 

def eval_clusters(clusters, graph):
    """
    Receives the predicted clusters and the gold standard graph and evaluates (with coref metrics) the predicate
    coreferences
    :param clusters: the predicted clusters
    :param graph: the gold standard graph
    :return: the predicate coreference metrics and the number of singletons
    """
    graph1_ent_mentions = []
    graph2_ent_mentions = clusters

    # Get the gold standard clusters
    for prop in graph.propositions.values():
        mentions_to_consider = set([mention for mention in prop.mentions.values() if mention.indices != [-1]])
        if len(mentions_to_consider) > 0:
            graph1_ent_mentions.append(mentions_to_consider)

    graph1_ent_mentions = [set(map(str, cluster)) for cluster in graph1_ent_mentions]
    graph2_ent_mentions = [set(map(str, cluster)) for cluster in graph2_ent_mentions]

    # Evaluate
    muc1, bcubed1, ceaf1 = muc(graph1_ent_mentions, graph2_ent_mentions), \
                           bcubed(graph1_ent_mentions, graph2_ent_mentions), \
                           ceaf(graph1_ent_mentions, graph2_ent_mentions)
    mela1 = np.mean([muc1, bcubed1, ceaf1])

    singletons = len([cluster for cluster in graph1_ent_mentions if len(cluster) == 1])
    return np.array([muc1, bcubed1, ceaf1, mela1]), singletons


   
def main():
    graphs = load_graphs_from_folder('../../data/baseline/test')
    scores = evaluate_predicate_coref(graphs)
    print(scores)


if __name__ == '__main__':
    main()