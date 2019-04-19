import astor
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))

from common.registerable import Registrable
from datasets.conala.utils import canonicalize_intent, decanonicalize_code, tokenize_intent
from asdl.lang.py.py_asdl_helper import asdl_ast_to_python_ast


class ExampleProcessor(object):
    """
    Process a raw input utterance using domain-specific procedures (e.g., stemming),
    and post-process a generated hypothesis to the final form
    """

    def pre_process_utterance(self, utterance):
        raise NotImplementedError

    def post_process_hypothesis(self, hyp, meta_info, **kwargs):
        raise NotImplementedError


@Registrable.register('conala_example_processor')
class ConalaExampleProcessor(ExampleProcessor):
    def __init__(self, transition_system):
        self.transition_system = transition_system

    @staticmethod
    def pre_process_utterance(utterance):
        canonical_intent, slot_map = canonicalize_intent(utterance)
        intent_tokens = tokenize_intent(canonical_intent)

        return intent_tokens, slot_map

    def post_process_hypothesis(self, hyp, meta_info, utterance=None):
        """traverse the AST and replace slot ids with original strings"""
        hyp_ast = asdl_ast_to_python_ast(hyp.tree, self.transition_system.grammar)
        code_from_hyp = astor.to_source(hyp_ast).strip()
        hyp.code = decanonicalize_code(code_from_hyp, meta_info)
