import os
import re
from collections import defaultdict
import sys
sys.path.append('../../../common')
sys.path.append('../..')

from eval_predicate_coref_changed import *
from okr import *
from clustering_common import cluster_mentions




def analyse_baseline_prop_coref(gold_annotations_dir = '../../data/baseline/dev', verbose = False):
	"""
	Qualitatively analyse the output from baseline for predicate coreference system. This
	method outputs all sentences, then list of gold predicate clusters and the list of predicted predicate clusters.
	"""
	parser = spacy_wrapper()
	for file_name in os.listdir(gold_annotations_dir):
		
		graph = load_graph_from_file(gold_annotations_dir + '/' + file_name)

		prop_mentions = []
		
		for prop in graph.propositions.values():
			for mention in prop.mentions.values():

				if mention.indices == [-1]:
					continue
				head_lemma, head_pos = get_mention_head(mention, parser, graph)
				prop_mentions.append((mention, head_lemma, head_pos))

		predicted_clusters = cluster_mentions(prop_mentions, score)
		
		sentence_wise_predicted_mentions = defaultdict(list)
		sentence_wise_gold_mentions = defaultdict(list)
		
		# Mappings 
		mention_to_gold_pred = {}
		mention_to_predicted_pred = {}

		# gold_dist -> Gold distribution of each predicted proposition cluster 
		# predicted_dist -> Predicted distribution of each gold proposition cluster
		
		gold_dist = {} 
		predicted_dist = {}

		for prop in graph.propositions.values():
			for mention in prop.mentions.values():
				sentence_wise_gold_mentions[mention.sentence_id].append({"indices":mention.indices,"coref":prop.id, 'text':mention.terms, 'arguments':mention.argument_mentions.values()})
				mention_to_gold_pred[str(mention)] = prop.id
				

		for prop_id, prop in enumerate(predicted_clusters):
			gold_dist[prop_id] = defaultdict(list)
			for mention, mention_head, _ in prop:
				sentence_wise_predicted_mentions[mention.sentence_id].append({"indices":mention.indices,"coref":prop_id, 'text':mention.terms})
				mention_to_predicted_pred[str(mention)] = prop_id
				gold_pred = mention_to_gold_pred[str(mention)]
				gold_dist[prop_id][gold_pred].append((mention, mention_head))

		for prop in graph.propositions.values():
			predicted_dist[prop.id] = defaultdict(list)
			for mention in prop.mentions.values():
				if(str(mention) in mention_to_predicted_pred.keys()):
					predicted_predicate = mention_to_predicted_pred[str(mention)]
					predicted_dist[prop.id][predicted_predicate].append(mention)	
				else:
					predicted_dist[prop.id][-1].append(mention)				
						

		print '###'+ file_name + '\n'	

		for sentence_id, sentence in enumerate(graph.sentences.values()):
			print 'Sentence: ', str(sentence_id + 1),' ' , ' '.join(sentence) 
			print 'Gold predicates: ', ', '.join([element['text'] + '(' + str(element['coref']) + ')' + '{' + ', '.join([str(argument) for argument in element["arguments"]]) + '}'  for element in sentence_wise_gold_mentions[sentence_id+1]])
			print 'Predicted predicates: ', ', '.join([element['text'] + '(' + str(element['coref']) + ')'  for element in sentence_wise_predicted_mentions[sentence_id+1]])
			print ' '

		print('\n The Gold clusters:')	

		for prop_id in predicted_dist.keys():
			print 'cluster_id: ', prop_id
			for predicted_pred_id in predicted_dist[prop_id].keys():
				print "\npredicted id:", predicted_pred_id, ':', ', '.join([mention.terms + '(' + str(mention.sentence_id) + ')' for mention in predicted_dist[prop_id][predicted_pred_id]])



		print 'The predicted clusters:'

		for prop_id in gold_dist.keys():
			print 'cluster_id: ', prop_id
			for gold_pred_id in gold_dist[prop_id].keys():
				print "\ngold id:", gold_pred_id, ':', ', '.join([mention.terms + '('+ mention_head + ', '+ str(mention.sentence_id) + ')' for mention,mention_head in gold_dist[prop_id][gold_pred_id]])



		print '**********'



def create_report_for_prop_coref(gold_annotations_dir = '../../data/baseline/dev'):
	"""
	Qualitatively analyse the output from baseline for predicate coreference system. This method is similar to 
	analyse_baseline_prop_coref() but only displays the distribution of the gold and predicted clusters for which 
	the system has made mistakes in clustering (two kinds of mistakes - incorrect merges and missed merges). Only sample
	representatives of a cluster are displayed. 
	"""
	parser = spacy_wrapper()
	for file_name in os.listdir(gold_annotations_dir):
		
		graph = load_graph_from_file(gold_annotations_dir + '/' + file_name)

		prop_mentions = []
		
		for prop in graph.propositions.values():
			for mention in prop.mentions.values():

				if mention.indices == [-1]:
					continue
				head_lemma, head_pos = get_mention_head(mention, parser, graph)
				prop_mentions.append((mention, head_lemma, head_pos))

		predicted_clusters = cluster_mentions(prop_mentions, score)
		
		sentence_wise_predicted_mentions = defaultdict(list)
		sentence_wise_gold_mentions = defaultdict(list)
		
		# Mappings 
		mention_to_gold_pred = {}
		mention_to_predicted_pred = {}

		# Gold distribution of each predicted prop
		gold_dist = {} 
		predicted_dist = {}

		for prop in graph.propositions.values():
			for mention in prop.mentions.values():
				sentence_wise_gold_mentions[mention.sentence_id].append({"indices":mention.indices,"coref":prop.id, 'text':mention.terms, 'arguments':mention.argument_mentions.values()})
				mention_to_gold_pred[str(mention)] = prop.id
				

		for prop_id, prop in enumerate(predicted_clusters):
			gold_dist[prop_id] = defaultdict(list)
			for mention, mention_head, _ in prop:
				sentence_wise_predicted_mentions[mention.sentence_id].append({"indices":mention.indices,"coref":prop_id, 'text':mention.terms})
				mention_to_predicted_pred[str(mention)] = prop_id
				gold_pred = mention_to_gold_pred[str(mention)]
				gold_dist[prop_id][gold_pred].append((mention, mention_head))

		for prop in graph.propositions.values():
			predicted_dist[prop.id] = defaultdict(list)
			for mention in prop.mentions.values():
				if(str(mention) in mention_to_predicted_pred.keys()):
					predicted_predicate = mention_to_predicted_pred[str(mention)]
					predicted_dist[prop.id][predicted_predicate].append(mention)	
				else:
					predicted_dist[prop.id][-1].append(mention)				
						

		print '###'+ file_name + '\n'	


		print('Mentions which should have been clustered: ')
		for prop_id in predicted_dist.keys():
			
			if(len(predicted_dist[prop_id].keys())!=1):
				print 'gold_cluster_id: ', prop_id
				mentions_to_print = [predicted_dist[prop_id][predicted_pred_id][0] for predicted_pred_id in predicted_dist[prop_id].keys() ]
				print (', '.join([mention.terms + '(' + str(mention.sentence_id) + ')' +  
					'{' + ', '.join([' '.join([graph.sentences[mention.sentence_id][int(id)] 
						for id in str(argument).rstrip(']').split('[')[1].split(', ')     ]) 
					for argument in mention.argument_mentions.values()]) + '}' 
					for mention in mentions_to_print]))


				
		print("\nMentions which should not have been clustered: ")
		for prop_id in gold_dist.keys():
			if(len(gold_dist[prop_id])!=1):
				print 'predicted_cluster_id: ', prop_id
				mentions_to_print = [gold_dist[prop_id][gold_pred_id][0][0] for gold_pred_id in gold_dist[prop_id].keys() ]
				print (', '.join([mention.terms + '(' + str(mention.sentence_id) + ')' +  
					'{' + ', '.join([' '.join([graph.sentences[mention.sentence_id][int(id)] 
						for id in str(argument).rstrip(']').split('[')[1].split(', ')     ]) 
					for argument in mention.argument_mentions.values()]) + '}' 
					for mention in mentions_to_print]))




		

		print '**********'



        


def main():
	# analyse_baseline_prop_coref(verbose = True)
	create_report_for_prop_coref()

if __name__ == '__main__':
		main()	