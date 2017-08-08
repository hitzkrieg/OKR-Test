"""
Do proposition coreference using argument coreference (predicted argument coreference and gold predicate extraction) 
Author: Hitesh Golchha

Coref score achieved by best system :  [0.62, 0.74, 0.56, 0.64]

""" 



import sys
sys.path.append('../common')
sys.path.append('../agreement')

import numpy as np

from okr import *
from entity_coref import *
from clustering_common import cluster_mentions
from parsers.spacy_wrapper import spacy_wrapper



import spacy
from munkres import *
from fuzzywuzzy import fuzz
from spacy.en import English
from num2words import num2words
from nltk.corpus import wordnet as wn
from eval_entity_coref import similar_words, same_synset, fuzzy_fit, partial_match, is_stop


# Don't use spacy tokenizer, because we originally used NLTK to tokenize the files and they are already tokenized
nlp = spacy.load('en')



def mention_string_to_terms(graph, mention, sentence_id):
	"""
	Returns the string of terms given a mention, the sentence id and the OKR object. This function is used 
	in this file specially argument mentions.
	:param graph: the OKR object 
	:param mention: mention object whose terms need to be found.
	:sentence id: Sentence id of the sentence to which the mention belongs.
	"""
	terms = ' '.join([graph.sentences[sentence_id][int(id)] for id in str(mention).rstrip(']').split('[')[1].split(', ')     ])
	return terms


def evaluate_predicate_coref(test_graphs, lexical_wt, argument_match_ratio):
	"""
	Receives the OKR test graphs and evaluates them for predicate coreference
	:param test_graphs: the OKR test graphs
	:param lexical_wt: the lexical weight for the scoring function which compares two proposition clusters. (the argument weight is 1- lexical_wt)
	:param argument_match_ratio: the minimum argument alignment threshold between propositions
	:return: the coreference scores: MUC, B-CUBED, CEAF and MELA (CoNLL F1).
	"""

	scores = []

	for graph in test_graphs:

	   
		args_and_entities_clusters = get_argument_and_entity_clusters(graph)
		clusters = get_proposition_clusters(graph, args_and_entities_clusters, lexical_wt, argument_match_ratio)
		
		# Evaluate
		curr_scores, _ = eval_clusters(clusters, graph)
		scores.append(curr_scores)

	scores = np.mean(scores, axis=0).tolist()

	return scores




def get_argument_and_entity_clusters(graph):
	"""
	Extract entities and argument mentions from the graphs, take their union and cluster using the baseline entity corefeence algorithm.
	:param graph: the OKR objec
	"""
	entities = [(str(mention), unicode(mention.terms)) for entity in graph.entities.values() for mention in entity.mentions.values()]

	arguments = [(str(mention), unicode(mention_string_to_terms(graph, mention, prop_mention.sentence_id ))) for prop in graph.propositions.values() for prop_mention in prop.mentions.values() for mention in prop_mention.argument_mentions.values() if prop_mention.indices!=[-1]]
	args_and_entities_union = list(set(entities).union(set(arguments)))
	args_and_entities_clusters = cluster_mentions(args_and_entities_union, argument_score)
	args_and_entities_clusters = [set([item[0] for item in cluster]) for cluster in args_and_entities_clusters]
	return args_and_entities_clusters

def get_proposition_clusters(graph, args_and_entities_clusters, lexical_wt, argument_match_ratio):
	"""
	Finds the proposition clusters obtained after proposition coreference step. Uses the cluster of arguments and entities
	also for decision making.
	:param graph: the OKR object
	:param args_and_entities_clusters: cluster of arguments and entities obtained from coreference
	:param lexical_wt: the lexical weight for the scoring function. (the argument weight is 1- lexical_wt)
	:param argument_match_ratio: the minimum argument alignment threshold between propositions
	:return: the proposition clusters: clusters
	"""

	prop_mentions = []
	parser = spacy_wrapper()

	for prop in graph.propositions.values():
		for mention in prop.mentions.values():

			if mention.indices == [-1]:
				continue

			head_lemma, head_pos = get_mention_head(mention, parser, graph)
			prop_mentions.append((mention, head_lemma, head_pos))

	clusters = cluster_prop_mentions(prop_mentions, score_prime, args_and_entities_clusters, lexical_wt, argument_match_ratio)
	clusters = [set([item[0] for item in cluster]) for cluster in clusters]
	return clusters

	



def cluster_prop_mentions(mention_list, score, argument_clusters, lexical_wt, argument_match_ratio):
	"""
	Cluster the predicate mentions in a greedy way: assign each predicate to the first
	cluster with similarity score > 0.5. If no such cluster exists, start a new one.
	:param mention_list: the mentions to clustert
	:param score: the score function that receives a mention and a cluster and returns a score
	:param clusters: the initial clusters received by the algorithm from a previous coreference pipeline
	:return: clusters of mentions
	"""
	clusters = []

	for mention in mention_list:
		found_cluster = False
		for cluster in clusters:
			if score(mention, cluster, argument_clusters, lexical_wt, argument_match_ratio) > 0.5:
				cluster.add(mention)
				found_cluster = True
				break

		if not found_cluster:
			clusters.append(set([mention]))

	return clusters








def argument_score(mention, cluster):
	"""
	
	:param mention: the mention
	:param cluster: the cluster
	:return: a numeric value denoting the similarity between the mention to the cluster
	"""
	return len([other for other in cluster if similar_words(other[1], mention[1])]) / (1.0 * len(cluster))



def score_prime(prop, cluster, argument_clusters, lexical_wt, argument_match_ratio):
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
	argument_score = len([other for other in cluster if (some_arg_match(other[0],prop[0], argument_clusters, argument_match_ratio ) )]) / (1.0 * len(cluster))
	
	return lexical_wt * lexical_score + (1-lexical_wt)*argument_score



def first_arg_match(prop_mention1, prop_mention2, argument_clusters):
	"""
	Finds if the first arguments of two propositions are coreferent.
	:param prop_mention1: First proposition mention
	:param prop_mention2: Second proposition mention
	:param argument_clusters: The clusters of arguments obtained by coreference
	:return: True if the first arguments of two propositions are coreferent.

	"""
	for m_id1, arg_mention1 in prop_mention1.argument_mentions.iteritems():
		for m_id2, arg_mention2 in prop_mention2.argument_mentions.iteritems():
			
			for cluster in argument_clusters:
				if str(arg_mention1) in cluster:
					if str(arg_mention2) in cluster:
						return True
					else:
						return False

	

def some_arg_match(prop_mention1, prop_mention2, argument_clusters, arg_match_ratio):
	"""
	Finds if atleast some % of arguments of two propositions are coreferent.
	:param prop_mention1: First proposition mention
	:param prop_mention2: Second proposition mention
	:param argument_clusters: The clusters of arguments obtained by coreference
	:arg_match_ratio: criteria of argument overlap = no of aligned arguments / no of arguments(minimum of proposition1 and proposition2)
	:return: True if atleast arg_match_ratio of arguments of two propositions are coreferent.

	"""
	matched_arguments = 0
	for m_id1, arg_mention1 in prop_mention1.argument_mentions.iteritems():
		pair_found = False
		for m_id2, arg_mention2 in prop_mention2.argument_mentions.iteritems():
			for cluster in argument_clusters:
				if str(arg_mention1) in cluster and str(arg_mention2) in cluster:
					matched_arguments+=1
					pair_found = True
					break
			if pair_found == True:
				break					

	if(matched_arguments == 0):
		return False
	elif(len(prop_mention1.argument_mentions)+ len(prop_mention2.argument_mentions)- matched_arguments==0):
		print 'problem here:'
		print 'Matched arguments: ', matched_arguments
		print 'No of arguments in the two propositions:' , len(prop_mention1.argument_mentions), ', ', len(prop_mention2.argument_mentions)
		print 'Proposition 1:', prop_mention1
		print 'Proposition 2:', prop_mention2
		print 'Proposition1 arguments: ', [arg_mention1 for m_id1, arg_mention1 in prop_mention1.argument_mentions.iteritems()]
		print 'Proposition 2 arguments: ',[arg_mention2 for m_id2, arg_mention2 in prop_mention2.argument_mentions.iteritems()]
		return False
		
	else:
		return (matched_arguments/ (len(prop_mention1.argument_mentions)+ len(prop_mention2.argument_mentions)- matched_arguments)>=arg_match_ratio)        



def some_word_match(string1, string2):
	"""
	Finds if any non-stop word in string1 is similar to any word in string2.
	:param string1: First string
	:param string2: Second string
	:return: True/False
	"""
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




def grid_search_for_parameters():
	"""
	Do a grid search over the parameters lexical_wt and argument_match_ratio on the dev set to choose best settings.
	"""

	print "Using predicted Entity coreference "
	graphs = load_graphs_from_folder('../../data/baseline/dev')

	lexical_wts = [0.75]
	argument_match_ratios = [0.5, 0.75, 1.0]

	for lexical_wt in lexical_wts:
		for argument_match_ratio in argument_match_ratios:
			scores = evaluate_predicate_coref(graphs, lexical_wt, argument_match_ratio)
			scores = [float("{:.2f}".format(x)) for x in scores] 
			print 'lexical_wt: {} argument_match_ratio: {} score: {}'.format(lexical_wt, argument_match_ratio, scores)

 
def main():
	"""
	print scores for the best setting of parameters.
	"""
	print "Using predicted Entity coreference "
	graphs = load_graphs_from_folder('../../data/baseline/test')

	lexical_wt = 0.75
	argument_match_ratio = 0.75

	scores = evaluate_predicate_coref(graphs, lexical_wt, argument_match_ratio)
	scores = [float("{:.2f}".format(x)) for x in scores] 
	print 'lexical_wt: {} argument_match_ratio: {} score: {}'.format(lexical_wt, argument_match_ratio, scores)


if __name__ == '__main__':
		main()    
