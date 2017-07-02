import os
import re


input_folder = '../../../src/common/sentences/dev'
output_folder ='../../../src/baseline_system/openie_extractions/dev'

# input_folder = '../common/sentences'
# output_folder = './openie_extractions'

for f in os.listdir(input_folder):

	#if f is car_bomb.txt, name becomes car_bomb
	f_name = re.match(r'(.+).txt', f).group(1)

	# Create a directory by the name of the output folder if it does not exist originally
	if(os.path.isdir(output_folder) == False):
		os.system('mkdir {}'.format(output_folder))
	os.system('sbt "runMain edu.knowitall.openie.OpenIECli --input-file {}/{} --format column --output-file {}/{}.prop" '.format(input_folder, f, output_folder, f_name))
