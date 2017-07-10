from okr import *

gold = load_graph_from_file('../../data/baseline/test/car_bomb.xml')

# Get argument mentions


gold_arg_mentions_dicts = { prop_id : [{ m_id : str(mention)
                                           for m_id, mention in mention.argument_mentions.iteritems()}
                                         for mention in prop.mentions.values()]
                              for prop_id, prop in gold.propositions.iteritems() }


gold_arg_mentions = { p_id : [set([mention_dict[str(arg_num)]
                                     for mention_dict in mention_lst if str(arg_num) in mention_dict])
                                for arg_num in range(0, 10)]
                        for p_id, mention_lst in gold_arg_mentions_dicts.iteritems()}

gold_arg_mentions = {k: [s for s in v if len(s) > 0] for k, v in gold_arg_mentions.iteritems()}


print(gold_arg_mentions[0])