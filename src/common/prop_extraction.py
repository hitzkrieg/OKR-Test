"""
Usage:
    prop_extraction --in=INPUT_FILE --out=OUTPUT_FILE

Extract propositions from a given input file, output is produced in separate output file.
If both in and out parmaters are directories, the script will itereate over all *.txt files in the input directory and
output to *.prop files in output directory.

Options:
   --in=INPUT_FILE    The input file, each sentence in a separate line
   --out=OUTPUT_FILE  The output file, Each extraction in a tab separated line, each consisting of original sentence,
   predicate template, lemmatized predicate template,argument name, argument value, ...

Author: Gabi Stanovsky
"""
import os
import sys
sys.path.append('../')

import logging

from spacy.en import English
from spacy.tokens import Span
from collections import defaultdict

import ntpath
import codecs
import logging

from glob import glob
from docopt import docopt

logging.basicConfig(level = logging.INFO)

class spacy_wrapper:
    """
    Abstraction over the spaCy parser, all output uses word indexes. Also offers VP and NP chunking as spaCy primitives.
    """
    def __init__(self):
        self.nlp = English()
        self.idx_to_word_index = {}
         
    def parse(self, sent):
        """
        Parse a raw sentence - shouldn't return a value, but properly change the internal status
        :param sent - a raw sentence
        """
        self.toks = self.nlp(unicode(sent, errors = 'ignore'))
        self.idx_to_word_index = self.get_idx_to_word_index()

    def get_sents(self):
        """
        Returns a list of sentences
        :return: a list of sentences
        """
        return [s for s in self.parser.sents]
        
    def chunk(self):
        """
        Run all chunking on the current sentence
        """
        self.np_chunk()
        self.vp_chunk()
        self.pp_chunk()

    def np_chunk(self):
        """
        spaCy noun chunking, will appear as single token
        See: https://github.com/explosion/spaCy/issues/156
        """
        for np in self.toks.noun_chunks:
            np.merge(np.root.tag_, np.text, np.root.ent_type_)
            
        # Update mappings
        self.idx_to_word_index = self.get_idx_to_word_index()

    def vp_chunk(self):
        """
        Verb phrase chunking - head is a verb and children are auxiliaries
        """
        self.chunk_by_filters(head_filter = lambda head: self.is_verb(head),
                              child_filter = lambda child: self.is_aux(child) and len(self.get_children(child)) == 0)
        
    def pp_chunk(self):
        """
        PP phrase chunking - head is a PP with a single PP child
        """
        def pp_head_filter(head):
            if not self.is_prep(head): return False
            children = self.get_children(head)
            if len(children) != 1: return False
            return self.is_prep(children[0])
                
        self.chunk_by_filters(head_filter = pp_head_filter, child_filter = lambda child: self.is_prep(child))

    def chunk_by_filters(self, head_filter, child_filter):
        """
        Meta chunking function, given head and children filters, collapses them together.
        Both head_filter and child_filter are functions taking a single argument - the node index.
        :param head_filter: head filter
        :param child_filter: child filter
        """

        # Collect verb chunks - dictionary from verb index to chunk's word indices '''
        chunks = defaultdict(lambda: [])
        for head in [i for i, _ in enumerate(self.toks) if head_filter(i)]:
            for child in [child for child in self.get_children(head) if child_filter(child)]:
                chunks[head].append(child)

        # Create Spans
        for head, span in chunks.iteritems():

            # The head itself is always part of a non-empty span
            span.append(head)
            span = sorted(span)
            if not consecutive(span):
                logging.warn("Assumption broken - chunk's children form a consecutive span: {}".format(span))

            # Calculate and store the span's lemma
            span_lemma = ' '.join([self.get_lemma(tok) for tok in span])
            chunks[head] = (Span(self.toks, span[0], span[-1] + 1), span_lemma)

        # Merge Spans to chunks, this is done after creating all Span objects to avoid index changing while iterating
        for head, (chunk, chunkLemma) in chunks.iteritems():

            # set the span's lemma
            chunk.merge(chunk.root.tag_, chunkLemma, chunk.root.ent_type_) # tag, lemma, ent_type

        # Update mappings
        self.idx_to_word_index = self.get_idx_to_word_index()

    def get_idx_to_word_index(self):
        """
        Create a mapping from idx to word index
        :return: a mapping from idx to word index
        """
        return dict([(tok.idx, i) for (i, tok) in enumerate(self.toks)])

    def get_pos(self, ind):
        """
        Return the pos of token at index ind
        :param ind: the index
        :return: the pos of token at index ind
        """
        return self.toks[ind].tag_

    def get_rel(self, ind):
        """
        Return the dependency relation of token at index ind
        :param ind: the index
        :return: the dependency relation of token at index ind
        """
        return self.toks[ind].dep_

    def get_word(self, ind):
        """
        Return the surface form of token at index ind
        :param ind: the index
        :return: the surface form of token at index ind
        """
        return self.toks[ind].orth_

    def get_lemma(self, ind):
        """
        Return the surface form of token at index ind
        :param ind: the index
        :return: the surface form of token at index ind
        """
        return self.toks[ind].lemma_
    
    def get_head(self, ind):
        """
        Return the word index of the head of of token at index ind
        :param ind: the index
        :return: the word index of the head of of token at index ind
        """
        return self.idx_to_word_index[self.toks[ind].head.idx]

    def get_children(self, ind):
        """
        Return a sorted list of children of a token
        :param ind: the index
        :return: a sorted list of children of a token
        """
        # TODO: do we have to use sorted here? Maybe spaCy already returns a sorted list of children
        return sorted([self.idx_to_word_index[child.idx] for child in self.toks[ind].children])

    def get_char_start(self, ind):
        """
        Get the start character index of this word
        :param ind: the index
        :return: the start character index of this word
        """
        return self.toks[ind].idx

    def get_char_end(self, ind):
        """
        Get the end character index of this word
        :param ind: the index
        :return: the end character index of this word
        """
        return self.toks[ind].idx + len(self.get_word(ind))
        
    def is_root(self, ind):
        """
        Returns True iff the token at index ind is the head of this tree
        :param ind: the index
        :return: True iff the token at index ind is the head of this tree
        """
        return (self.toks[ind].head is self.toks[ind])
    
    def get_len(self):
        """
        Returns the number of tokens in the current sentence
        :return: the number of tokens in the current sentence
        """
        return len(self.toks)

    def is_verb(self, ind):
        """
        Returns whether this token is a verb
        :param ind: the index
        :return: Is this token a verb
        """
        return self.get_pos(ind).startswith('VB')

    def is_pronoun(self, ind):
        """
        Returns whether this token is a pronoun
        :param ind: the index
        :return: Is this token a pronoun
        """
        return self.get_pos(ind).startswith('WP')

    def is_aux(self, ind):
        """
        Returns whether this token is a auxiliary
        :param ind: the index
        :return: Is this token a auxiliary
        """
        return self.get_rel(ind).startswith('aux')

    def is_dative(self, ind):
        """
        Returns whether this token is a dative
        :param ind: the index
        :return: Is this token a dative
        """
        return self.get_rel(ind).startswith('dative')

    def is_prep(self, ind):
        """
        Returns whether this token is a preposition
        :param ind: the index
        :return: Is this token a preposition
        """
        return self.get_rel(ind).startswith('prep')

    def is_subj(self, ind):
        """
        Returns whether this token is a subject
        :param ind: the index
        :return: Is this token a subject
        """
        return 'subj' in self.get_rel(ind)

    def is_obj(self, ind):
        """
        Returns whether this token is a object
        :param ind: the index
        :return: Is this token a object
        """
        rel = self.get_rel(ind)
        return ('obj' in rel) or ('attr' in rel)

    def is_rel_clause(self, ind):
        """
        Returns whether this token is a relative clause
        :param ind: the index
        :return: Is this token a relative clause
        """
        return 'relcl' in self.get_rel(ind)

    def get_single_pobj(self, ind):
        """
        Get Pobj, only if there's exactly one such child
        :param ind: the index
        :return: Pobj, only if there's exactly one such child
        """
        pobjs = [child for child in self.get_children(ind) if self.get_rel(child) == 'pobj']
        if len(pobjs) == 1:
            return pobjs

        # TODO: what to do if there's zero or more than one pobj?
        return []

    def get_text(self, ind):
        """
        Get the text of this node
        :param ind: the index
        :return: the text of this node
        """
        return self.toks[ind].text

        
def consecutive(span):
    """
    Check if a span of indices is consecutive
    :param span: the span of indices
    :return: whether the span of indices is consecutive
    """
    return [i - span[0] for i in span] == range(span[-1] - span[0] + 1)


def main():

    args = docopt(__doc__)
    inp = args['--in']
    out = args['--out']
    logging.info("Loading spaCy...")
    pe = prop_extraction()

    # Differentiate between single input file and directories
    if os.path.isdir(inp):
        logging.debug('Running on directories:')
        num_of_lines = num_of_extractions = 0
        for input_fn in glob(os.path.join(inp, '*.txt')):
            output_fn = os.path.join(out, path_leaf(input_fn).replace('.txt', '.prop'))
            logging.debug('input file: {}\noutput file:{}'.format(input_fn, output_fn))

            cur_line_counter, cur_extractions_counter = run_single_file(input_fn, output_fn, pe)
            num_of_lines += cur_line_counter
            num_of_extractions += cur_extractions_counter

    else:
        logging.debug('Running on single files:')
        num_of_lines, num_of_extractions = run_single_file(inp, out, pe)

    logging.info('# Sentences: {} \t #Extractions: {} \t Extractions/sentence Ratio: {}'.
                 format(num_of_lines, num_of_extractions, float(num_of_extractions) / num_of_lines))


class prop_extraction:
    """
    Lenient proposition extraction -- assumes all modifications are non-restrictive
    """
    def __init__(self):
        """
        Initalize internal parser
        """
        self.parser = spacy_wrapper()

    def get_extractions(self, sent):
        """
        Given a sentence, get all of its propositions.
        :param sent: the sentence
        :return A list of strings, each representing a single proposition.
        """
        ret = []

        # Special formatting of output for OKR agreement purposes
        okr_ret = []

        try:
            self.parser.parse(sent)

            for verb in [i for i in range(self.parser.get_len()) if self.parser.is_verb(i)]:

                # Extract the proposition from this predicate
                extraction = Extraction()

                # OKR additions:
                okr_pred_indices = []
                okr_pred_terms = ""

                # For each child, decide if and how to include it in the template
                for child in self.parser.get_children(verb):

                    curr_arg = child

                    # Find if this is the right spot to plug the verb in the template
                    if (curr_arg > verb) and (not extraction.pred_exists):

                        extraction.set_predicate(self.parser.get_text(verb), self.parser.get_lemma(verb))

                        # Take care of OKR stuff
                        okr_pred_indices.append(verb)
                        okr_pred_terms += "{} ".format(self.parser.get_text(verb))

                    # Datives and prepositions are deferred if they have one exactly one pobj
                    if self.parser.is_dative(curr_arg) or self.parser.is_prep(curr_arg):

                        pobjs = self.parser.get_single_pobj(curr_arg)

                        if pobjs:

                            # Sanity check
                            assert(len(pobjs)) == 1
                            pobj = pobjs[0]

                            # Plug prep/dative in template and signal that pobj should be added to rolesDict
                            # note - we do not lemmatize the datives and pp's
                            extraction.template += "{} ".format(self.parser.get_text(curr_arg))

                            # Take care of OKR stuff
                            okr_pred_indices.append(curr_arg)
                            okr_pred_terms += "{} ".format(self.parser.get_text(curr_arg))

                            curr_arg = pobj

                    # Subject and objects are plugged directly
                    if self.parser.is_subj(curr_arg):

                        # Replace pronoun with head of a relative clause
                        if self.parser.is_rel_clause(verb) and self.parser.is_pronoun(curr_arg):
                            extraction.add_argument(self.parser.get_text(self.parser.get_head(verb)))
                        else:
                            extraction.add_argument(self.parser.get_text(curr_arg))

                    if self.parser.is_obj(curr_arg):
                        extraction.add_argument(self.parser.get_text(curr_arg))

                # Record extractions with at least 2 arguments
                if len(extraction.args) > 1 and extraction.pred_exists:
                    ret.append(str(extraction))

                # Record OKR predicates:
                if okr_pred_indices:
                    okr_pred_terms = okr_pred_terms.strip()
                    if len(okr_pred_indices) != len(okr_pred_terms.split(' ')):

                        # Sanity check -- this happens due to Spacy's chunking which messes up indexes
                        logging.warn("Length of indices and pred differ: {} != {}".format(okr_pred_indices, okr_pred_terms))
                    okr_ret.append((okr_pred_indices, okr_pred_terms))
        except:
            raise

        return okr_ret


class Extraction:
    """
    A representation of an extraction, composed of a single predicate and an arbitrary (>0) number of arguments.
    """
    def __init__(self):
        self.args = []
        self.roles_dict = {}
        self.template = ''
        self.pred_exists = False

    def set_predicate(self, pred_text, lemmatized_pred_text):
        """
        Add the predicate and its lemmatized version to this template
        :param pred_text: the predicate template
        """
        self.template += '{} '.format(pred_text)
        self.pred = pred_text
        self.lemmatized_pred = lemmatized_pred_text
        self.pred_exists = True

    def add_argument(self, arg_text):
        """
        Add a new argument to the extraction
        Adds to template and roles_dict, by assuming the next index after the last argument.
        :param arg_text: the argument text
        """
        arg_index = len(self.args)
        self.args.append(arg_text)
        self.roles_dict[arg_index] = arg_text  # Note: This ignores all internal modifications
        self.template += '{A' + str(arg_index) + '} '

    def __str__(self):
        """
        Textual representation of this extraction for an output file.
        """
        self.template = self.template.lstrip().rstrip()

        # Create a lemmatized template, by replacing the predicate slot with its lemma
        self.lemmatized_template = self.template.replace(self.pred, self.lemmatized_pred)
        ret = '\t'.join([self.template, self.lemmatized_template] +
                        ['A{}\t{}'.format(key, val)
                         for key, val in sorted(self.roles_dict.iteritems(), key = lambda (k,_): k)])
        return ret


def run_single_file(input_fn, output_fn, prop_ex):
    """
    Process extractions from a single input file and print to an output file,
    using a proposition extraction module.
    :param input_fn: the input file name
    :param output_fn: the output file name
    :param prop_ex: the proposition extraction object
    :return (#lines, #num of extractions)
    """
    logging.info("Reading sentences from {}".format(input_fn))
    ex_counter = 0
    line_counter = 0

    with codecs.open(output_fn, 'w', 'utf-8') as f_out:

        for line in open(input_fn):
            line_counter += 1
            data = line.strip().split('\t')
            tweet_id = None
            if len(data) == 2:
                tweet_id, sent = data
            else:
                # Not at tweet, just fill in the id with a place holder
                tweet_id = 'NONE'
                sent = data[0]
            logging.info('Read: {}'.format(sent))
            for ex in prop_ex.get_extractions(sent):
                to_print = '\t'.join(map(str, [tweet_id, sent, ex])).decode('ascii', errors = 'ignore')
                f_out.write(to_print + "\n")
                ex_counter += 1

    logging.info('Done! Wrote {} extractions to {}'.format(ex_counter, output_fn))
    return (line_counter, ex_counter)


def path_leaf(path):
    """
    Get just the filename from the full path.
    http://stackoverflow.com/questions/8384737/extract-file-name-from-path-no-matter-what-the-os-path-format
    """
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


if __name__ == '__main__':
    main()

