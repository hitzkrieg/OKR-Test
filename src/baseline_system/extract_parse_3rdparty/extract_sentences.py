"""
Usage:
    extract_sentences --in=INPUT_FOLDER --out=OUTPUT_FOLDER

Extract sentences from all xml filesin the input folder, output is produced in individual files in OUTPUT_FOLDER.
(requires some minor debugging I think)


Options:
   --in=INPUT_FOLDER    The input folder
   --out=OUTPUT_FOLDER  The output folder

"""
import os
import copy
import codecs
from docopt import docopt
import re
import xml.etree.ElementTree as ET

def main():
    print('Hi')
    args = docopt(__doc__)
    inp = args['--in']
    out = args['--out']
    extract_sentences_from_folder(inp, out)


def extract_sentences_from_folder(input_folder, output_folder):
    # print('Hi')
	for f in os.listdir(input_folder):
		extract_sentences_from_file(input_folder + "/" + f, output_folder) 

def extract_sentences_from_file(input_file, output_folder):
        # print('Hi')

    # Load the xml to a tree object
    tree = ET.parse(input_file)

    # Load the sentences
    root = tree.getroot()
    sentences_node = root.find('sentences')[1:]

    # Handle different versions - old version:
    if sentences_node[0].find('str') != None:
        sentences = { int(sentence.find('id').text): sentence.find('str').text.split() for sentence in sentences_node }
        ignored_indices = None
        tweet_ids = {}

    # New version
    else:
        sentences = { int(sentence.find('id').text) : [token.find('str').text for token in sentence.find('tokens')]
                     for sentence in sentences_node }
        ignored_indices = set(
            [sentence.find('id').text + '[' + token.find('id').text + ']' for sentence in sentences_node
             for token in sentence.find('tokens') if token.find('isIrrelevant').text == 'true'])
        tweet_ids = {int(sentence.find('id').text): sentence.find('name').text for sentence in sentences_node}

   	output_file = output_folder + "/" + re.match(r'(.+).xml',input_file).group(1) + ".sent"

   
   	with codecs.open(output_file, 'w', 'utf-8') as f_out:
 		for sent in sentences.values():
   			f_out.write(' '.join(sent))


if __name__ == '__main__':
    main()
