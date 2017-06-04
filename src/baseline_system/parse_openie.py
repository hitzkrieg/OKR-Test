import re
import os

# Regex for matching argument and relation elements.
regex1 = re.compile(r"([a-zA-Z]+)\((.+),List\((.*)\)") 
# Output from Openie 4.0


for file_name in os.listdir('./openie_extractions'):
	with open("./openie_extractions/{}".format(file_name)) as output_from_openie:
	# output_from_openie = open("output_raw_sentences.txt")

		for line in output_from_openie:
			
			# args_char_indices = []
			# relation_char_indices = []
			# context_char_indices = []
			# simple_argument_char_indices
			
			splitted_line = line.split('\t')
			confidence = float(splitted_line[0])
			original_sentence = splitted_line[-1].strip()
			rest = splitted_line[2:-1]
			
			if(splitted_line[1]!= ''):
				if(splitted_line[1]==None):
					print('None instead of space in Context portion')
					print(splitted_line)
				else:		
					match1 = regex1.search(splitted_line[1])
					if(match1 == None):
						print('************')
						print(splitted_line[1])
					else:	
						indices = list(map(int, re.findall(r"\d+", match1.group(3))))
						context_indices = [(indices[2*i], indices[2*i +1]) for i in range(int(len(indices)/2))]
						context_text = match1.group(2)
				
			count_arg = 0 
			count_tem_arg = 0
			count_spa_arg = 0
			count_rel = 0

			for element in rest:
				if( len(element.split('; ')) > 1):
					for i in element.split('; '):
						rest.append(i)
					continue
				# print(element)	
				match = regex1.search(element)
				if(match!= None) and (match.group(1) == 'SimpleArgument'):
					
					simple_argument_text = match.group(2)
					indices = list(map(int, re.findall(r"\d+", match.group(3))))
					simple_argument_indices = [(indices[2*i], indices[2*i +1]) for i in range(int(len(indices)/2))]
					count_arg+=1
					# assert simple_argument_text == 
					#TO DO: assert

				elif(match!= None) and (match.group(1) == 'Relation'):
					
					
					relation_text = match.group(2)
					indices = list(map(int, re.findall(r"\d+", match.group(3))))
					relation_char_indices = [(indices[2*i], indices[2*i +1]) for i in range(int(len(indices)/2))]
					count_rel+=1
					#TO DO: assert


				elif(match!= None) and (match.group(1) == 'SpatialArgument'):
					
					
					spatial_argument_text = match.group(2)
					indices = list(map(int, re.findall(r"\d+", match.group(3))))
					spatial_argument_char_indices = [(indices[2*i], indices[2*i +1]) for i in range(int(len(indices)/2))]
					count_spa_arg+=1
					#TO DO: assert	

					

				elif(match!= None) and (match.group(1) == 'TemporalArgument'):
					temporal_argument_text = match.group(2)
					indices = list(map(int, re.findall(r"\d+", match.group(3))))
					temporal_argument_char_indices = [(indices[2*i], indices[2*i +1]) for i in range(int(len(indices)/2))]
					count_tem_arg+=1
					#TO DO: assert


				elif(match!= None):
					

					print('***********************')
					print('Not identified (similar format): {}, splitted_line {}'.format(element, splitted_line))
					
					#Identify other kind of output.

						

				else:
					
					if(element!=''):
						print('***********************')
						print('Not identified (different format):{}, splitted_line: {}'.format(element, splitted_line))
					#Identify other kind of output.


			if(count_rel >1 ):
				print('Relations: {}'.format(count_rel))
			if(count_arg >1 ):
				print('Arguments: {}'.format(count_arg))
			if(count_tem_arg >1 ):
				print('TemporalArguments: {}'.format(count_tem_arg))
			if(count_spa_arg >1 ):
				print('SpatialArguments: {}'.format(count_spa_arg))			