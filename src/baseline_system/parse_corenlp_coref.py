import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
import sys
sys.path.append('../common')

from eval_entity_coref import *
from okr import *
"""
It could have been possible to have just reuse the existing code for evaluation, but I consider to convert the CoreNLP
output into the ECB like format and then try to analyse from there.
"""




def parse_corenlp_coref_xml_doc(input_dir = 'CoreNLP_coref_anno/dev'):
	"""
	Parses the output xml files in the input diretory annotataed by the coreNLP and outputs it in a form 
	similar to ECB+ in a file named as 'coref_output.txt'.  
	"""

	mentions = []
	for file in os.listdir(input_dir):
		tree = ET.parse(input_dir + '/' + file)
		document = tree.getroot()[0]
		# sentences_node = document.find('sentences')

		# for sentence in enumerate(sentences_node):
		# 	s_num = sentence.attribs['id']
		# 	sentence_text = " ".join([token.word for token in sentence.find('tokens')])
		# 	sentences[s_num] = sentence_text

		coref_node = document.find('coreference')
		
		for coref_id, coref_chain in enumerate(coref_node):
			for mention in cluster:
				sent_num = int(mention[0].text)
				start = int(mention[1].text)-1
				end = int(mention[2].text)-1
				text = mention[4].text
				mentions.append({"filename":file, "s_num":sent_num,"EP":"E", "indices":range(start, end),"coref":coref_id+1})

	mentions.sort(key=lambda x:(x["filename"],x["s_num"],x["indices"][0]))
	with open('coref_output.txt', 'w') as out_file:
		out_file.write("file\tsentence\tentity(E) or predicate(P)\t coref chain\tindices\t\n")
		out_file.write("\n".join([e["filename"]+"\t"+str(e["s_num"])+"\t"+e["EP"]+"\t"+str(e["coref"])+"\t"+str(e["indices"])[1:-1] for e in mentions]))


def parse_and_evaluate_corenlp_coref(input_dir = 'CoreNLP_coref_anno/dev', gold_annotations_folder = '../../data/baseline/dev'): 
	"""
	Parse the output xml file annotated by coreNLP and evaluate the accuracy of mentions and coreference resolution with 
	gold annotations.
	"""
	
	scores = []
	
	for file in os.listdir(input_dir):
		if re.match(r'(.+)\.xml', file)!= None:
			clusters = []
			okr_graph = load_graph_from_file(gold_annotations_folder + '/'+ re.match(r'(.+)\.xml', file).group(1)[:-4]+'.xml')
			tree = ET.parse(input_dir + '/' + file)
			document = tree.getroot()[0]
			coref_node = document.find('coreference')
			
			for coref_id, coref_chain in enumerate(coref_node):
				cluster = []
				for mention in coref_chain:
					sent_num = int(mention[0].text)
					start = int(mention[1].text)-1
					end = int(mention[2].text)-1
					indices = range(start,end)
					text = mention[4].text
					mention_string = str(sent_num)+ str(indices)
					cluster.append((mention_string, text))
				clusters.append(cluster)
			clusters = [set([item[0] for item in cluster]) for cluster in clusters]
		
			# gold_mentions = [set(map(str, entity.mentions.values())) for entity in okr_graph.entities.values()]
			# print('\n')
			# print('Gold Mentions:', gold_mentions)

			# print('*********')
			# print('*********')
			curr_scores = eval_clusters(clusters, okr_graph)
	        scores.append(curr_scores)

	print(scores)		
	scores = np.mean(scores, axis=0).tolist()    
	print(scores)


def parse_and_analyse_corenlp_coref(input_dir = 'CoreNLP_coref_anno/dev', gold_annotations_folder = '../../data/baseline/dev'):
	"""
	Sample examples, look and evaluate qualitatively where we are making mistakes for CoreNLP
	"""
	mentions = []


	with open('coref_analyse_output.txt', 'w') as out_file:

		for file_name in os.listdir(input_dir):
			if re.match(r'(.+)\.xml', file_name)!= None:
				okr_graph = load_graph_from_file(gold_annotations_folder + '/'+ re.match(r'(.+)\.xml', file_name).group(1)[:-4]+'.xml')

				tree = ET.parse(input_dir + '/' + file_name)
				document = tree.getroot()[0]
				sentence_wise_predicted_mentions = defaultdict(list)
				sentence_wise_gold_mentions = defaultdict(list)
				predicted_coref_dict = defaultdict(list)
				gold_coref_dict = defaultdict(list)

				coref_node = document.find('coreference')
				
				
				for coref_id, coref_chain in enumerate(coref_node):
					for mention in coref_chain:
						sent_num = int(mention[0].text)
						start = int(mention[1].text)-1
						end = int(mention[2].text)-1
						text = mention[4].text
						sentence_wise_predicted_mentions[sent_num].append({"indices":range(start, end),"coref":coref_id+1, "text":text})
						predicted_coref_dict[coref_id+1].append({"indices":range(start, end), "s_num":sent_num, "text":text })


				
								
				for entity in okr_graph.entities.values():
					for mention in entity.mentions.values():
						sentence_wise_gold_mentions[mention.sentence_id].append({"indices":mention.indices,"coref":entity.id, 'text':mention.terms})

				print'###'+ file_name + '\n'	
				for sentence_id, sentence in enumerate(okr_graph.sentences.values()):
					print 'Sentence: ', ' '.join(sentence) 
					print 'Predicted entities: ', [element['text'] for element in sentence_wise_predicted_mentions[sentence_id+1]]
					print 'Gold entities: ', [element['text'] for element in sentence_wise_gold_mentions[sentence_id+1]]
					print ' '
			
				print "Not printing singletons"
				print('\nThe predicted clusters: ')
				for cluster_id, cluster in enumerate(predicted_coref_dict.values()):
					print('Cluster id: ', cluster_id +1)
					print([[okr_graph.sentences[mention['s_num']][index] for index in mention['indices']]for mention in predicted_coref_dict[cluster_id+1]] )

				print('\n The Gold clusters:')	
				for entity in okr_graph.entities.values():
					print('cluster_id: ', entity.id )
					print([mention.terms for mention in entity.mentions.values()])

				print '**********'	
				
def main():
	parse_and_analyse_corenlp_coref()




if __name__ == '__main__':
	main()
