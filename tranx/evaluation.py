# coding=utf-8

import random
import sys
import traceback

from tqdm import tqdm

import common.cli_logger as cli_logger


## TODO: create decoder for each dataset
def decode(examples, model, args, shuffle=False, verbose=False, **kwargs):
    """
    TODO: add doc
    """

    if verbose:
        cli_logger.debug(f'[decode] evaluating {len(examples)} examples')

    if shuffle:
        cli_logger.debug(f'[decode] shuffling examples')
        random.shuffle(examples)

    was_training = model.training
    model.eval()

    decoded_results = []

    for example in tqdm(examples, desc='Decoding', file=sys.stdout, total=len(examples)):
        if verbose: cli_logger.debug(example)

        # get all hypothesis from the model, sorted by the score
        hyps = model.parse(example.src_sent, context=None, beam_size=args.beam_size)
        decoded_hyps = []

        for i, hyp in enumerate(hyps):
            got_code = False

            try:
                hyp.code = model.transition_system.ast_to_surface_code(hyp.tree)
                if verbose: cli_logger.debug(hyp.code)

                got_code = True
                decoded_hyps.append(hyp)
            except Exception as e:
                cli_logger.exception(e)

                if verbose:
                    err_msg = 'Exception in converting tree to code:\n' + '-' * 64 + '\n' + \
                              'Example: %s\nIntent: %s\nTarget Code:\n%s\nHypothesis[%d]:\n%s' % (
                                  example.idx, " ".join(example.src_sent), example.tgt_code, i, hyp.tree.to_string()
                              )
                    cli_logger.error(err_msg, do_raise=False)

                    if got_code: cli_logger.debug("\nGot code:" + hyp.code)

                    traceback.print_exc(file=sys.stdout)
                    cli_logger.debug('-' * 60)

        decoded_results.append(decoded_hyps)

    if was_training:
        model.train()

    return decoded_results


def evaluate(examples, model, evaluator, args, verbose=False, return_decoded_result=False, eval_top_pred_only=False):
    cli_logger.info(f"[evaluate] Decoding ...")
    decoded_results = decode(examples, model, args, shuffle=False, verbose=verbose)

    cli_logger.info(f"[evaluate] Evaluate decoded results ...")
    eval_result = evaluator.evaluate_dataset(examples, decoded_results, fast_mode=eval_top_pred_only)

    if return_decoded_result:
        return eval_result, decoded_results
    else:
        return eval_result
