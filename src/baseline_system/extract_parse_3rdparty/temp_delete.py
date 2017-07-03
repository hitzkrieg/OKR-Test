import os

input_dir = './coref_sentence_wise/dev'
gold_annotations_folder = '../../../data/baseline/dev'
import re



for file in os.listdir(gold_annotations_folder):
		if re.match(r'(.+)\.xml', file)!= None:
			for annotated_file in os.listdir(input_dir):
				if(re.match(file[:-4] + r'\.txt\.sent(\d+)\.txt.xml', annotated_file)!= None):
					print annotated_file