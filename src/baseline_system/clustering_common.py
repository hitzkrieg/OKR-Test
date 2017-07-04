"""
Utility script for clustering functions
"""


def cluster_mentions(mention_list, score, clusters = [], merge_initially= False):
    """
    Cluster the predicate mentions in a greedy way: assign each predicate to the first
    cluster with similarity score > 0.5. If no such cluster exists, start a new one.
    :param mention_list: the mentions to cluster
    :param score: the score function that receives a mention and a cluster and returns a score
    :param clusters: the initial clusters received by the algorithm from a previous coreference pipeline
    :return: clusters of mentions
    """

    if(merge_initially == True):
        if(len(clusters)!=0):
            for cluster1 in clusters:
                for cluster2 in clusters:
                    if cluster2!= cluster1:
                        if(len([mention1 for mention1 in cluster1 if score(mention1, cluster2)> 0.5]) +  len([mention2 for mention2 in cluster2 if score(mention2, cluster1)> 0.5])   > (len(cluster1) + len(cluster2)) /3.0):
                            # union of sets
                            cluster_merged = cluster1 | cluster2 

                            # Don't know why but sometimes error was appearing: cannot remove cluster because it doesnt exist
                            if(cluster1 in clusters):
                                clusters.remove(cluster1)
                            if(cluster2 in clusters):
                                clusters.remove(cluster2)
                            clusters.append(cluster_merged)


    for mention in mention_list:
        found_cluster = False
        for cluster in clusters:
            if score(mention, cluster) > 0.5:
                cluster.add(mention)
                found_cluster = True
                break

        if not found_cluster:
            clusters.append(set([mention]))

    return clusters





