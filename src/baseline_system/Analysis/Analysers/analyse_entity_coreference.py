import os
import re
from collections import defaultdict
import sys
sys.path.append('../../../common')
sys.path.append('../..')


from eval_entity_coref import *
from okr import *
from clustering_common import cluster_mentions

"""
Qualitatively analyse the output from baseline for entity coreference system
"""

def visually_analyse_baseline_entity_coref(gold_annotations_dir = '../../data/baseline/dev' ):
	
	for file_name in os.listdir(gold_annotations_dir):
		
		graph = load_graph_from_file(gold_annotations_dir + '/' + file_name)
		
		entities = [(str(mention), unicode(mention.terms)) for entity in graph.entities.values() for mention in
                    entity.mentions.values()]
		predicted_clusters = cluster_mentions(entities, score)

		sentence_wise_predicted_mentions = defaultdict(list)
		sentence_wise_gold_mentions = defaultdict(list)
	


		for entity in graph.entities.values():
			for mention in entity.mentions.values():
				sentence_wise_gold_mentions[mention.sentence_id].append({"indices":mention.indices,"coref":entity.id, 'text':mention.terms})


		# for entity_id, entity in enumerate(predicted_clusters):
		# 	for mention_id, mention_terms in entity:
		# 		indices = map(int, mention_id.split('[')[1].rstrip(']').split(', '))
		# 		sentence_wise_predicted_mentions[int(mention_id.split('[')[0])].append({"indices":indices,"coref":entity_id, 'text':mention_terms})		

		

		print'###'+ file_name + '\n'	
		for sentence_id, sentence in enumerate(graph.sentences.values()):
			print 'Sentence: ', ' '.join(sentence) 
			# print 'Predicted entities: ', ', '.join([element['text'] for element in sentence_wise_predicted_mentions[sentence_id+1]])
			print 'Gold entities: ', ', '.join([element['text'] for element in sentence_wise_gold_mentions[sentence_id+1]])
			print ' '

		print('\n The Gold clusters:')	
		for entity in graph.entities.values():
			if len(entity.mentions.values())!=1:
				print'cluster_id: ', entity.id , ', '.join([mention.terms for mention in entity.mentions.values()])
		print ' '

		print 'The predicted clusters:'
		for entity_id, entity in enumerate(predicted_clusters):
			if len(entity)!=1:
				print 'cluster_id: ', entity_id, ', '.join([mention_terms for _, mention_terms in entity])
		print ' '



		print '**********'

				

def generate_list_of_errors(gold_annotations_dir = '../../data/baseline/dev'):
	for file_name in os.listdir(gold_annotations_dir):
		
		graph = load_graph_from_file(gold_annotations_dir + '/' + file_name)
		
		entities = [(str(mention), unicode(mention.terms)) for entity in graph.entities.values() for mention in
                    entity.mentions.values()]
		predicted_clusters = cluster_mentions(entities, score)

		sentence_wise_predicted_mentions = defaultdict(list)
		sentence_wise_gold_mentions = defaultdict(list)

		# Mappings 
		mention_to_gold_entity = {}
		mention_to_predicted_entity = {}

		# Gold distribution of each predicted prop
		gold_dist_of_predicted_entities = {} 
		predicted_dist_of_gold_entities = {}

		for entity in graph.entities.values():
			for mention in entity.mentions.values():
				sentence_wise_gold_mentions[mention.sentence_id].append({"indices":mention.indices,"coref":entity.id, 'text':mention.terms})
				mention_to_gold_entity[str(mention)] = entity.id

		for entity_id, entity in enumerate(predicted_clusters):
			gold_dist_of_predicted_entities[entity_id] = defaultdict(list)
			for mention_id, mention_terms in entity:
				sentence_wise_predicted_mentions[mention_id].append({"coref":entity_id, 'text':mention_terms})
				
				mention_to_predicted_entity[mention_id] = entity_id
				gold_entity = mention_to_gold_entity[mention_id]
				gold_dist_of_predicted_entities[entity_id][gold_entity].append((mention_id, mention_terms))		


		for entity in graph.entities.values():
			predicted_dist_of_gold_entities[entity.id] = defaultdict(list)
			for mention in entity.mentions.values():
				if(str(mention) in mention_to_predicted_entity.keys()):
					predicted_entity = mention_to_predicted_entity[str(mention)]
					predicted_dist_of_gold_entities[entity.id][predicted_entity].append(mention)	
				else:
					predicted_dist_of_gold_entities[entity.id][-1].append(mention)					


		print'###'+ file_name + '\n'	

		print('Mentions which should have been clustered: ')
		for entity_id in predicted_dist_of_gold_entities.keys():
			
			if(len(predicted_dist_of_gold_entities[entity_id].keys())!=1):
				print 'gold_cluster_id: ', entity_id
				mentions_to_print = [predicted_dist_of_gold_entities[entity_id][predicted_entity_id][0] for predicted_entity_id in predicted_dist_of_gold_entities[entity_id].keys() ]
				# print mentions_to_print
				if((mentions_to_print != None )):
					print (', '.join([mention.terms for mention in mentions_to_print]))

		print("\nMentions which should not have been clustered: ")
		for entity_id in gold_dist_of_predicted_entities.keys():
			if(len(gold_dist_of_predicted_entities[entity_id])!=1):
				print 'predicted_cluster_id: ', entity_id
				mentions_to_print = [gold_dist_of_predicted_entities[entity_id][gold_entity_id][0] for gold_entity_id in gold_dist_of_predicted_entities[entity_id].keys() ]
				if((mentions_to_print != None )):				
					print (', '.join([mention_terms for _, mention_terms in mentions_to_print]))		


		

		# for sentence_id, sentence in enumerate(graph.sentences.values()):
		# 	print 'Sentence: ', ' '.join(sentence) 
		# 	print 'Gold entities: ', ', '.join([element['text'] for element in sentence_wise_gold_mentions[sentence_id+1]])
		# 	print ' '

		# print('\n The Gold clusters:')	
		# for entity in graph.entities.values():
		# 	if len(entity.mentions.values())!=1:
		# 		print'cluster_id: ', entity.id , ', '.join([mention.terms for mention in entity.mentions.values()])
		# print ' '

		# print 'The predicted clusters:'
		# for entity_id, entity in enumerate(predicted_clusters):
		# 	if len(entity)!=1:
		# 		print 'cluster_id: ', entity_id, ', '.join([mention_terms for _, mention_terms in entity])
		# print ' '



		print '**********'	

						

def main():
	generate_list_of_errors()
if __name__ == '__main__':
		main()	