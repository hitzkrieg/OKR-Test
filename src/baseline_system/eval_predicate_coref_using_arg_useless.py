"""
Series of experiments to try to leverage entity coreference for predicate coreference.
In this file we use gold proposition extractions as well as 
Author: Hitesh Golchha
""" 
import sys
sys.path.append('../common')
sys.path.append('../agreement')

import numpy as np

from okr import *
from entity_coref import *
from parsers.spacy_wrapper import spacy_wrapper



import spacy
from munkres import *
from fuzzywuzzy import fuzz
from spacy.en import English
from num2words import num2words
from nltk.corpus import wordnet as wn


# Don't use spacy tokenizer, because we originally used NLTK to tokenize the files and they are already tokenized
nlp = spacy.load('en')

def is_stop(w):
    return w in spacy.en.STOP_WORDS


def evaluate_predicate_coref_using_everything_gold(test_graphs, arg_match_ratio, lexical_vs_argument_ratio):
    """
    Receives the OKR test graphs and evaluates them for predicate coreference by leveraging entity coreference; using the entity mentions,  
    predicate mentions, and argument coreference from gold
    :param test_graphs: the OKR test graphs
    :param arg_match_ratio: the 
    :param lexical_vs_argument_ratio: 
    :return: the coreference scores: MUC, B-CUBED, CEAF and MELA (CoNLL F1).
    """

    parser = spacy_wrapper()

    scores = []

    for graph in test_graphs:
        prop_mentions = []

        gold_arg_mentions_dicts = { prop_id : [{ m_id : str(mention)
                                               for m_id, mention in mention.argument_mentions.iteritems()}
                                             for mention in prop.mentions.values()]
                                  for prop_id, prop in graph.propositions.iteritems() }

        # Clusters of arguments per proposition
        gold_arg_mentions = { p_id : [set([mention_dict[str(arg_num)]
                                             for mention_dict in mention_lst if str(arg_num) in mention_dict])
                                        for arg_num in range(0, 10)]
                                for p_id, mention_lst in gold_arg_mentions_dicts.iteritems()}

        # Remove empty arguments
        gold_arg_mentions = {k: [s for s in v if len(s) > 0] for k, v in gold_arg_mentions.iteritems()}


        for prop in graph.propositions.values():
            for mention in prop.mentions.values():

                if mention.indices == [-1]:
                    continue

                head_lemma, head_pos = get_mention_head(mention, parser, graph)
                prop_mentions.append((mention, head_lemma, head_pos))

        clusters = cluster_mentions(prop_mentions, score_prime, gold_arg_mentions, arg_match_ratio, lexical_vs_argument_ratio)
        clusters = [set([item[0] for item in cluster]) for cluster in clusters]    
        # Evaluate
        curr_scores, _ = eval_clusters(clusters, graph)
        scores.append(curr_scores)

    scores = np.mean(scores, axis=0).tolist()

    return scores






def score(prop, cluster, gold_arg_mentions):
    return len([other for other in cluster if atleast_half_arg_match(other[0],prop[0], gold_arg_mentions)]) / (1.0 * len(cluster))

def some_arg_match(prop_mention1, prop_mention2, gold_arg_mentions, arg_match_ratio):
    matched_arguments = 0
    for m_id1, arg_mention1 in prop_mention1.argument_mentions.iteritems():
        for m_id2, arg_mention2 in prop_mention2.argument_mentions.iteritems():
            for p_id, mention_list in gold_arg_mentions.iteritems():
                for argument_set in mention_list:
                    if str(arg_mention1) in argument_set:
                        if str(arg_mention2) in argument_set:
                            matched_arguments+=1

    return (matched_arguments >= min(len(prop_mention1.argument_mentions) , len(prop_mention2.argument_mentions))* arg_match_ratio)

                        

def first_arg_match(prop_mention1, prop_mention2, gold_arg_mentions):
    for m_id1, arg_mention1 in prop_mention1.argument_mentions.iteritems():
        for m_id2, arg_mention2 in prop_mention2.argument_mentions.iteritems():
            for p_id, mention_list in gold_arg_mentions.iteritems():
                for argument_set in mention_list:
                    if str(arg_mention1) in argument_set:
                        if str(arg_mention2) in argument_set:
                            # print 'Hello'
                            return True
                        else:
                            return False

    return False                        


def score_prime(prop, cluster, gold_arg_mentions, arg_match_ratio, lexical_vs_argument_ratio):
    """
    Receives a proposition mention (mention, head_lemma, head_pos)
    and a cluster of proposition mentions, and returns a numeric value denoting the
    similarity between the mention to the cluster (% of same head lemma mentions in the cluster)
    :param prop: the mention
    :param cluster: the cluster
    :return: a numeric value denoting the similarity between the mention to the cluster
    """
    
    # return len([other for other in cluster if similar_words(other[1],prop[1])]) / (1.0 * len(cluster))
    lexical_score = len([other for other in cluster if (some_word_match(other[0].terms,prop[0].terms))]) / (1.0 * len(cluster))
    argument_score = len([other for other in cluster if (some_arg_match(other[0],prop[0], gold_arg_mentions , arg_match_ratio) )]) / (1.0 * len(cluster))
    
    return lexical_vs_argument_ratio * lexical_score + (1- lexical_vs_argument_ratio)*argument_score




def some_word_match(string1, string2):
    words_list1 = string1.split(' ')
    words_list2 = string2.split(' ')
    intersection = []
    for a in words_list1:
        for b in words_list2:
            if(similar_words(a,b)):
                intersection.append(a)
                break

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


def get_distance_to_root(token, parser):
    """
    Receives a token and returns its distance from the root
    :param token: the token
    :param parser: the spacy wrapper object
    :return: the distance from the token to the root
    """
    dist = 0
    while parser.get_head(token) != token:
        token = parser.get_head(token)
        dist += 1
    return dist


def get_mention_head(mention, parser, graph):
    """
    Gets a mention and returns its head
    :param mention: the mention
    :param parser: the spacy wrapper object
    :param graph: the OKR graph
    :return: the mention head
    """
    distances_to_root = []
    curr_head_and_pos = []
    sentence = graph.sentences[mention.sentence_id]

    joined_sentence = ' '.join(sentence)
    parser.parse(joined_sentence)

    for index in mention.indices:
        child = parser.get_word(index)
        child_lemma = parser.get_lemma(index)
        child_pos = parser.get_pos(index)
        head = parser.get_word(parser.get_head(index))

        if parser.get_head(index) in mention.indices and head != child:
            continue

        distances_to_root.append(get_distance_to_root(index, parser))
        curr_head_and_pos.append((child_lemma, child_pos))

    # Get the closest to the root
    best_index = np.argmin(distances_to_root)
    curr_head, curr_pos = curr_head_and_pos[best_index]

    return curr_head, curr_pos

def similar_words(x, y):
    """
    Returns whether x and y are similar
    :param x: the first mention
    :param y: the second mention
    :return: whether x and y are similar
    """
    return same_synset(x, y) or fuzzy_fit(x, y) or partial_match(x, y) 


def same_synset(x, y):
    """
    Returns whether x and y share a WordNet synset
    :param x: the first mention
    :param y: the second mention
    :return: whether x and y share a WordNet synset
    """
    x_synonyms = set([lemma.lower().replace('_', ' ') for synset in wn.synsets(x) for lemma in synset.lemma_names()])
    y_synonyms = set([lemma.lower().replace('_', ' ') for synset in wn.synsets(y) for lemma in synset.lemma_names()])

    return len([w for w in x_synonyms.intersection(y_synonyms) if not is_stop(w)]) > 0


def fuzzy_fit(x, y):
    """
    Returns whether x and y are similar in fuzzy string matching
    :param x: the first mention
    :param y: the second mention
    :return: whether x and y are similar in fuzzy string matching
    """
    if fuzz.ratio(x, y) >= 90:
        return True

    # Convert numbers to words
    x_words = [num2words(int(w)).replace('-', ' ') if w.isdigit() else w for w in x.split()]
    y_words = [num2words(int(w)).replace('-', ' ') if w.isdigit() else w for w in y.split()]

    return fuzz.ratio(' '.join(x_words), ' '.join(y_words)) >= 85


def partial_match(x, y):
    """
    Return whether these two mentions have a partial match in WordNet synset.
    :param x: the first mention
    :param y: the second mention
    :return: Whether they are aligned
    """

    # Allow partial matching
    if fuzz.partial_ratio(' ' + x + ' ', ' ' + y + ' ') == 100:
        return True

    x_words = [w for w in x.split() if not is_stop(w)]
    y_words = [w for w in y.split() if not is_stop(w)]

    if len(x_words) == 0 or len(y_words) == 0:
        return False

    x_synonyms = [set([lemma.lower().replace('_', ' ') for synset in wn.synsets(w) for lemma in synset.lemma_names()])
                  for w in x_words]
    y_synonyms = [set([lemma.lower().replace('_', ' ') for synset in wn.synsets(w) for lemma in synset.lemma_names()])
                  for w in y_words]

    # One word - check whether there is intersection between synsets
    if len(x_synonyms) == 1 and len(y_synonyms) == 1 and \
                    len([w for w in x_synonyms[0].intersection(y_synonyms[0]) if not is_stop(w)]) > 0:
        return True

    # More than one word - align words from x with words from y
    cost = -np.vstack([np.array([len([w for w in s1.intersection(s2) if not is_stop(w)]) for s1 in x_synonyms])
                       for s2 in y_synonyms])
    m = Munkres()
    cost = pad_to_square(cost)
    indices = m.compute(cost)

    # Compute the average score of the alignment
    average_score = np.mean([-cost[row, col] for row, col in indices])

    if average_score >= 0.75:
        return True

    return False





def cluster_mentions(mention_list, score, gold_arg_mentions, arg_match_ratio, lexical_vs_argument_ratio):
    """
    Cluster the predicate mentions in a greedy way: assign each predicate to the first
    cluster with similarity score > 0.5. If no such cluster exists, start a new one.
    :param mention_list: the mentions to cluster
    :param score: the score function that receives a mention and a cluster and returns a score
    :param clusters: the initial clusters received by the algorithm from a previous coreference pipeline
    :return: clusters of mentions
    """
    clusters = []

    for mention in mention_list:
        found_cluster = False
        for cluster in clusters:
            if score(mention, cluster, gold_arg_mentions, arg_match_ratio, lexical_vs_argument_ratio) > 0.5:
                cluster.add(mention)
                found_cluster = True
                break

        if not found_cluster:
            clusters.append(set([mention]))

    return clusters






def main():
    print "hello!!!!"
    graphs = load_graphs_from_folder('../../data/baseline/test')

    ratios = [0.0, 0.25, 0.5, 0.75, 1.0]

    for lexical_vs_argument_ratio in ratios:
        for arg_match_ratio in ratios:
            scores = evaluate_predicate_coref_using_everything_gold(graphs, arg_match_ratio, lexical_vs_argument_ratio)
            print 'lexical_vs_argument_ratio: {} arg_match_ratio: {} score: {}'.format(lexical_vs_argument_ratio, arg_match_ratio, scores)

if __name__ == '__main__':
        main()    








