import re
import os
import sys


sys.path.append('../common')
from okr import *
from filter_propositions import*


# An implementation almost similar to verbal_filter from filter_propositions_propositions 
verbal_filter2 = lambda sentence, mention_indices: (len(mention_indices) == 1) and \
                                            (nltk.pos_tag(sentence)[mention_indices[0]][1].startswith('V'))



def character_indices_to_word_indices(character_indices, sentence):

	"""
	Convert the list of character range tuples extracted as an output of OpenIE4 and convert it into a list of word 
	indices suitable for okr
    :param character_indices: a list of tuples defining the range of character indices which form the mention
    :param sentence: the sentence string
    :return word_indices: list of word indices
    """

	tokenized_sentence = sentence.split(' ')
	token_wise_char_index = []
	word_indices = []
	# token_wise_char_index: The list of starting character of each token
	index_length=0
	for token in tokenized_sentence:
		token_wise_char_index.append(index_length)	
		index_length += len(token)+1

	for char_range in character_indices:	
		word_indices+=[i for i,_ in enumerate(tokenized_sentence) if token_wise_char_index[i]>=int(char_range[0]) and token_wise_char_index[i]< int(char_range[1])]

	return word_indices	


def parse_and_evaluate_openie(openie_extractions_folder = './openie_extractions/test', okr_graphs = load_graphs_from_folder('../../data/baseline/test')):

	"""
	Convert the output format of extractions from openIE into a form that can be used by the evaluation pipeline 
    :param openie_extractions_folder: folder consisting of text files, each of which is the output file from OpenIE4
    :return :TBD 
    """
    
	accuracy_all = 0
	accuracy_verbal = 0
	accuracy_non_verbal = 0
	accuracy_arg = 0

    # Regex for matching argument and relation elements.
	regex1 = re.compile(r"([a-zA-Z]+)\((.+),List\((.*)\)") 
	
	for file_name in os.listdir(openie_extractions_folder):
		print('File: {}'.format(file_name))
		flag = 0
		for graph in okr_graphs:
			if re.match(r'(.+)\.prop',file_name).group(1) in graph.name:
				flag=1
				break

		assert flag ==1

		sent_id = 1	
		with open(openie_extractions_folder+"/{}".format(file_name)) as output_from_openie:
		# output_from_openie = open("output_raw_sentences.txt")
			
			predicate_mentions = []
			argument_mentions = []
			verbal_predicate_mentions = []
			non_verbal_predicate_mentions = []
			
			for line in output_from_openie:

				splitted_line = line.split('\t')
				confidence = float(splitted_line[0])
				original_sentence = splitted_line[-1].strip()
				tokenized_sentence = original_sentence.split(' ')

				
				while(tokenized_sentence != graph.sentences[sent_id]):
					# print tokenized_sentence, graph.sentences[sent_id]
					sent_id+=1

					
			
				rest = splitted_line[2:-1]
				
				if(splitted_line[1]!= ''):
					if(splitted_line[1]==None):
						print('None instead of space in Context portion')
						print(splitted_line)
					else:		
						match1 = regex1.search(splitted_line[1])
						if(match1 == None):
							# Some unexpected form
							print('*****Unexpected*******')
							print(splitted_line[1])
						else:	
							indices = list(map(int, re.findall(r"\d+", match1.group(3))))
							context_char_indices = [(indices[2*i], indices[2*i +1]) for i in range(int(len(indices)/2))]
							context_word_indices = character_indices_to_word_indices(context_char_indices, original_sentence)
							context_text = match1.group(2)
							# if (context_text != ' '.join([tokenized_sentence[i] for i in context_word_indices])):
							# 	print('Context text found: {} \n Context text evaluated: {}\n\n'.format(context_text, ' '.join([tokenized_sentence[i] for i in context_word_indices])))
							# assert(context_text == ' '.join([tokenized_sentence[i] for i in context_word_indices]))

					
				count_arg = 0 
				count_rel = 0
				
				for element in rest:
					if( len(element.split('; ')) > 1):
						for i in element.split('; '):
							rest.append(i)
						continue
						
					match = regex1.search(element)
					if(match!= None) and (match.group(1) == 'SimpleArgument' or match.group(1) == 'SpatialArgument' or match.group(1) == 'TemporalArgument' ):
						
						argument_text = match.group(2)
						indices = list(map(int, re.findall(r"\d+", match.group(3))))
						argument_indices = [(indices[2*i], indices[2*i +1]) for i in range(int(len(indices)/2))]
						argument_word_indices = character_indices_to_word_indices(argument_indices, original_sentence)
						# if(argument_text != ' '.join([tokenized_sentence[i] for i in argument_word_indices])):
						# 	print("Argument text found: {} \n  argument text evaluated: {} \n ".format(argument_text, ' '.join([tokenized_sentence[i] for i in argument_word_indices])) )

						count_arg+=1

						argument_mentions.append(str(sent_id)+ str(argument_word_indices))


					elif(match!= None) and (match.group(1) == 'Relation'):
												
						relation_text = match.group(2)
						indices = list(map(int, re.findall(r"\d+", match.group(3))))
						relation_char_indices = [(indices[2*i], indices[2*i +1]) for i in range(int(len(indices)/2))]
						relation_word_indices = character_indices_to_word_indices(relation_char_indices, original_sentence)
						count_rel+=1
						# if(relation_text != ' '.join([tokenized_sentence[i] for i in relation_word_indices])):
						# 	print("Relation text found: {} \n  Relation evaluated: {} \n ".format(relation_text, ' '.join([tokenized_sentence[i] for i in relation_word_indices])) )

						assert(count_rel==1)
						# if(len(relation_word_indices)>0):
						predicate_mentions.append(str(sent_id)+ str(relation_word_indices))
						if(verbal_filter2(tokenized_sentence, relation_word_indices) == True):
							verbal_predicate_mentions.append(str(sent_id)+ str(relation_word_indices))
						else:
							non_verbal_predicate_mentions.append(str(sent_id)+ str(relation_word_indices))	


					elif(match!= None):
						#Identify other kind of output.
						print('**********Unexpected*************')
						print('Not identified (similar format): {}, splitted_line {}'.format(element, splitted_line))
						
					else:
						#Identify other kind of output.

						if(element!=''):
							print('*********Unexpected**************')
							print('Not identified (different format):{}, splitted_line: {}'.format(element, splitted_line))
						

		
			argument_mentions = set(argument_mentions)
			predicate_mentions = set(predicate_mentions)
			verbal_predicate_mentions = set(verbal_predicate_mentions)
			non_verbal_predicate_mentions = set(non_verbal_predicate_mentions)

			verbal_gold = filter_verbal(graph)
			non_verbal_gold = filter_non_verbal(graph)

			
			print('Evaluating predicate mentions:\n ')

			gold_graph_mentions = set.union(*[set(map(str, prop.mentions.values())) for prop in graph.propositions.values()])
			gold_graph_mentions_verbal = set.union(*[set(map(str, prop.mentions.values())) for prop in verbal_gold.propositions.values()])
			gold_graph_mentions_non_verbal =  set.union(*[set(map(str, prop.mentions.values())) for prop in non_verbal_gold.propositions.values()])

			gold_graph_argument_mentions = set.union(*[set.union(*[set([arg.str_p(mention) for arg in mention.argument_mentions.values()])
                                                  for mention in  prop.mentions.values()])
                                      for prop in graph.propositions.values()])


			a1 = evaluate_accuracy_from_graphs(gold_graph_mentions, predicate_mentions)
			a2 = evaluate_accuracy_from_graphs(gold_graph_mentions_verbal, verbal_predicate_mentions)
			a3 = evaluate_accuracy_from_graphs(gold_graph_mentions_non_verbal, non_verbal_predicate_mentions)

			a4 = evaluate_accuracy_from_graphs(gold_graph_argument_mentions, argument_mentions)
			print('Hello: {}'.format(a4))
    		accuracy_all += a1
    		accuracy_verbal += a2
    		accuracy_non_verbal += a3
    		accuracy_arg += a4

    		print('Accuracy all: {} Accuracy verbal: {} Accuracy non-verbal: {} '.format(a1, a2, a3))
    		print('Accuracy argument mentions: {}'.format(a4))

    		print('*********************************************************************\n\n')

    		


	accuracy_all = accuracy_all / len(okr_graphs)
	accuracy_verbal = accuracy_verbal / len(okr_graphs)
	accuracy_non_verbal = accuracy_non_verbal / len(okr_graphs)
	accuracy_arg = accuracy_arg / len(okr_graphs)

	print('Accuracy all: {} Accuracy verbal: {} Accuracy non-verbal: {} '.format(accuracy_all, accuracy_verbal, accuracy_non_verbal))
	print('Accuracy argument mentions: {}'.format(accuracy_arg))



# Given the list of mentions from the gold annotations and the predictions, evaluate the accuracy.
def evaluate_accuracy_from_graphs(gold_graph_mentions, predicate_mentions):
	common_sentences = set([x.split('[')[0] for x in gold_graph_mentions]).intersection(set([x.split('[')[0] for x in predicate_mentions]))

	gold_graph_mentions = set([a for a in gold_graph_mentions if a.split('[')[0] in common_sentences])
	predicate_mentions = set([a for a in predicate_mentions if a.split('[')[0] in common_sentences])

	consensual_mentions = gold_graph_mentions.intersection(predicate_mentions)

	accuracy1 = len(consensual_mentions) * 1.0 / len(gold_graph_mentions) if len(gold_graph_mentions) > 0 else 0.0
	accuracy2 = len(consensual_mentions) * 1.0 / len(predicate_mentions) if len(predicate_mentions) > 0 else 0.0
	return (   2.00 * (accuracy1 * accuracy2)/ (accuracy1 + accuracy2) if (accuracy1 + accuracy2) > 0.0 else 0.0     )
	# return ((accuracy1 + accuracy2) / 2.0)




def main():
	parse_and_evaluate_openie(openie_extractions_folder = './openie_extractions/testing/prop', okr_graphs = load_graphs_from_folder('./openie_extractions/testing/anno')) 



if __name__ == '__main__':
    main()
