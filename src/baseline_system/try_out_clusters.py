import sys

sys.path.append('../common')
sys.path.append('../agreement')


from okr import *
from eval_entity_coref import *

graph = load_graph_from_file('../../data/baseline/test/likud_beiteinu.xml')
entities = [(str(mention), unicode(mention.terms)) for entity in graph.entities.values() for mention in
                    entity.mentions.values()]
clusters = cluster_mentions(entities, score)
clusters = [set([item[0] for item in cluster]) for cluster in clusters]
cluster_id = 0


for cluster in clusters:
	for mention in cluster:
		sent_no, indices= mention.rstrp(']').split('[')
		 


cluster_id = 