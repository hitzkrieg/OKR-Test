"""
analyse_pred
Author: Hitesh Golchha

"""

import sys

sys.path.append('../../../common')
sys.path.append('../..')



from okr import *
from docopt import docopt
from eval_predicate_mention import *
from prop_extraction import prop_extraction


def main():
    """
    Receives the validation set and the test set, runs the baseline systems,
    and prints analysis logs of propositions
    """
    args = docopt("""Receives the validation set and the test set, runs the baseline systems,
    and prints analysis logs of propositions
    

    Usage:
        analyse_pred.py <val_set_folder> <test_set_folder>

        <val_set_folder> = the validation set file
        <test_set_folder> = the test set file
    """)

    val_folder = args['<val_set_folder>']
    test_folder = args['<test_set_folder>']

    # Load the annotation files to OKR objects
    val_graphs = load_graphs_from_folder(val_folder)
    test_graphs = load_graphs_from_folder(test_folder)

    # Load a common proposition extraction model
    logging.debug('Loading proposition extraction module')
    prop_ex = prop_extraction()

    logging.debug('Running analysis of predicates')
    # Run the predicate mentions component and evaluate them
    analyse_predicate_mentions(test_graphs, prop_ex, './nominalizations/nominalizations.reuters.txt')



if __name__ == '__main__':
    main()
