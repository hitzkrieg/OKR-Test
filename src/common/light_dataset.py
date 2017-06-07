import sys
from okr import *


graphs_dir=sys.argv[1]
graphs = load_graphs_from_folder(graphs_dir)
out_file=open("light_dataset_sentences","w")
out_file.write("file\tsentence\ttokens\n")
sentences=[{"filename":g.name[9:-4], "s_num":s_num, "tokens":" ".join(s)}for g in graphs for s_num,s in g.sentences.iteritems()]
sentences.sort(key=lambda x:(int(x["filename"]),x["s_num"]))
out_file.write("\n".join([s["filename"]+"\t"+str(s["s_num"])+"\t"+ s["tokens"] for s in sentences]))
out_file.close()
out_file=open("light_dataset_indices","w")
out_file.write("file\tsentence\tentity(E) or predicate(P)\t coref chain\tindices\t\n")
elements=[{"filename":g.name[9:-4],"s_num":m.sentence_id,"EP":"E","indices":m.indices, "coref":e_num} for g in graphs for e_num,e in g.entities.iteritems() for m in e.mentions.values()]
elements+=[{"filename":g.name[9:-4],"s_num":m.sentence_id,"EP":"P","indices":m.indices, "coref":p_num} for g in graphs for p_num,p in g.propositions.iteritems() for m in p.mentions.values() if m.is_explicit]
elements.sort(key=lambda x:(int(x["filename"]),x["s_num"],int(x["indices"][0])))

out_file.write("\n".join([e["filename"]+"\t"+str(e["s_num"])+"\t"+e["EP"]+"\t"+str(e["coref"])+"\t"+str(e["indices"])[1:-1] for e in elements]))
out_file.close()




