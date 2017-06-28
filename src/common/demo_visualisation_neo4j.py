from py2neo import Graph, Node, Relationship
from okr import *

# See documentation for how to use if default localhost port is not used or there is username/password configured
graph = Graph()


okr_graph1 = load_graph_from_file('../../data/baseline/test/car_bomb.xml')


for prop in okr_graph1.propositions.values():

	prop_node = Node("Proposition", id = prop.id, name = prop.name)
	graph.create(prop_node)

	for prop_mention in prop.mentions.values():

		prop_mention_node = Node("PropositionMention", terms = prop_mention.terms, sentence_id = prop_mention.sentence_id,
		 id = prop_mention.id, indices = prop_mention.indices)
		graph.create(prop_mention_node)

		graph.create(Relationship(prop_node, "has a mention", prop_mention_node))

		for argument in prop_mention.argument_mentions.values():

			argument_terms = ' '.join([okr_graph1.sentences[prop_mention.sentence_id][int(id)] for id in str(argument).rstrip(']').split('[')[1].split(', ')     ]) 
			argument_mention_node = Node("ArgumentMention", id = argument.id, desc = argument.desc, mention_type = argument.mention_type, terms = argument_terms)
			graph.create(Relationship(prop_mention_node, "has an argument mention", argument_mention_node))

for entity in okr_graph1.entities.values():
	entity_node = Node("Entity", id = entity.id, name = entity.name)
	graph.create(entity_node)

	for entity_mention in entity.mentions.values():

		entity_mention_node = Node("EntityMention", terms = entity_mention.terms, sentence_id = entity_mention.sentence_id, id = entity_mention.id, indices = entity_mention.indices)
		graph.create(entity_mention_node)
		graph.create(Relationship(entity_node, "has a mention", entity_mention_node))
