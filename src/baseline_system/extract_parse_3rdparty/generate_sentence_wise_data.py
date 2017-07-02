import os
import re







def generate_sentence_wise_data(input_dir = '../../common/sentences/dev', output_dir = '../../common/sentence_wise_files'):
	for file in os.listdir(input_dir):
		cover_directory = input_dir.split('/')[-1]
		os.system('mkdir {}'.format(output_dir + '/' + cover_directory))
		os.system('mkdir {}'.format(output_dir + '/' + cover_directory + '/' + file))
		with open(input_dir + '/' + file, 'r') as lines:
			lineno= 0
			for line in lines:
				lineno+=1
				if(line[-2]!= '.' or line[-2]!='!' or line[-2]!= '?'):
					line = line.rstrip('\n')+ ' .' +'\n'
				with open(output_dir + '/' + cover_directory + '/' + file + '/'+ 'sent' + str(lineno) + '.txt', 'w') as output_file:
					output_file.write(line)








def main():
	generate_sentence_wise_data()

if __name__ == '__main__':
	main()
