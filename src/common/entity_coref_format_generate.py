import sys

sys.path.append('../baseline_system')
sys.path.append('../agreement')


from okr import *
from eval_entity_coref import *

graphs_dir = '../../data/baseline/test'
graphs = load_graphs_from_folder(graphs_dir)

out_file=open("light_dataset_indices","w")
out_file.write("file\tsentence\tentity(E) or predicate(P)\t coref chain\tindices\t\n")
elements = []
cluster_id = 0
file_number = 1
for graph in graphs:
	file_number
	entities = [(str(mention), unicode(mention.terms)) for entity in graph.entities.values() for mention in
                    entity.mentions.values()]
	clusters = cluster_mentions(entities, score)
	clusters = [set([item[0] for item in cluster]) for cluster in clusters]                
	for cluster in clusters:
		for mention in cluster:
			sent_no, indices= mention.rstrip(']').split('[')
			elements.append({"filename":file_number,"s_num":sent_no,"EP":"E", "indices":indices, "coref": cluster_id})
		cluster_id+=1	
	file_number+=1	
		
elements.sort(key=lambda x:(file_number,int(x["s_num"]),int(x["indices"].split(', ')[0]) ))
elements.sort(key=lambda x:(file_number))
out_file.write("\n".join([str(e["filename"])+"\t"+str(e["s_num"])+"\t"+e["EP"]+"\t"+str(e["coref"])+"\t"+str(e["indices"]) for e in elements]))
out_file.close()
