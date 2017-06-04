"""
compute_baseline_subtasks
Author: Vered Shwartz

    Receives the validation set and the test set, runs the baseline systems,
    and computes the task-level evaluation metrics:
    1) Entity mentions
    2) Entity coreference
    3) Predicate mentions
    4) Predicate coreference
    5) Argument mention within predicate chains
    6) Entailment graph
"""

import sys

sys.path.append('../common')


from okr import *
from docopt import docopt
from eval_predicate_mention import *
from prop_extraction import prop_extraction
from eval_entity_coref import evaluate_entity_coref
from eval_entailment_graph import evaluate_entailment
from eval_argument_coref import evaluate_argument_coref
from eval_entity_mention import evaluate_entity_mention
from eval_predicate_coref import evaluate_predicate_coref
from eval_argument_mention import evaluate_argument_mention


def main():
    """
    Receives the validation set and the test set, runs the baseline systems,
    and computes the task-level evaluation metrics:
    1) Entity mentions
    2) Entity coreference
    3) Predicate mentions
    4) Predicate coreference
    5) Argument mention within predicate chains
    6) Entailment graph
    """
    args = docopt("""Receives the validation set and the test set, runs the baseline systems,
    and computes the task-level evaluation metrics:
    1) Entity mentions
    2) Entity coreference
    3) Predicate mentions
    4) Predicate coreference
    5) Argument mention within predicate chains
    6) Entailment graph

    Usage:
        compute_baseline_subtasks.py <val_set_folder> <test_set_folder>

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

    # Run the predicate mentions component and evaluate them
    analyse_predicate_mentions(test_graphs, prop_ex, './nominalizations/nominalizations.reuters.txt')



if __name__ == '__main__':
    main()
