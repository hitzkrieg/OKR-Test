import os
import re

def correct_end_of_tweet_punctuations_in_sentence_files(input_dir = '../../common/sentences', output_dir = '../../common/sentences_punctuated' ):
	for file in os.listdir(input_dir+ '/'+ 'dev'):
		with open(input_dir + '/dev/' + file, 'r') as input_filename: 
			with open(output_dir + '/dev/' + file, 'w') as output_filename:
				for line in input_filename:
					if(line[-2]!= '.' or line[-2]!='!' or line[-2]!= '?'):
						line = line.rstrip('\n')+ ' .' +'\n'
						output_filename.write(line)




if __name__ == '__main__':
	correct_end_of_tweet_punctuations_in_sentence_files()

