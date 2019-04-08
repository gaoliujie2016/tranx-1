# coding=utf-8

import argparse
import os
import sys
import time

import pickle
import torch
from tensorboardX import SummaryWriter

import common.cli_logger as cli_logger
import evaluation
from asdl.asdl_base import ASDLGrammar
from common.registerable import Registrable
from common.utils import dump_cfg, init_cfg
from components.dataset import Dataset
from model import nn_utils
from model.utils import GloveHelper
from model import parser


def prologue(cfg: argparse.Namespace, *varargs) -> SummaryWriter:
    # sanity checks
    assert cfg.exp_name not in [None, ""]
    assert not cfg.cuda or (cfg.cuda and torch.cuda.is_available())

    # dirs
    base_dir = f"./experiments/{cfg.exp_name}"

    os.makedirs(f"{base_dir}/out", exist_ok=True)
    os.makedirs(f"{base_dir}/chkpt", exist_ok=True)
    os.makedirs(f"{base_dir}/logs", exist_ok=True)

    dump_cfg(f"{base_dir}/train_config.txt", vars(cfg))

    # tb writer
    return SummaryWriter(f"{base_dir}/logs")


def epilogue(cfg: argparse.Namespace, *varargs) -> None:
    summary_writer = varargs[0]
    summary_writer.close()


# TODO: split logic into separate functions
def train(cfg: argparse.Namespace):
    cli_logger.info("=== Training ===")

    # initial setup
    summary_writer = prologue(cfg)

    # load train/dev set
    train_set = Dataset.from_bin_file(cfg.train_file)

    if cfg.dev_file:
        dev_set = Dataset.from_bin_file(cfg.dev_file)
    else:
        dev_set = Dataset(examples=[])

    vocab = pickle.load(open(cfg.vocab, 'rb'))

    grammar = ASDLGrammar.from_text(open(cfg.asdl_file).read())
    transition_system = Registrable.by_name(cfg.transition_system)(grammar)

    parser_cls = Registrable.by_name(cfg.parser)
    model = parser_cls(cfg, vocab, transition_system)
    model.train()

    evaluator = Registrable.by_name(cfg.evaluator)(transition_system, args=cfg)
    if cfg.cuda:
        model.cuda()

    optimizer_cls = eval(f'torch.optim.{cfg.optimizer}')
    optimizer = optimizer_cls(model.parameters(), lr=cfg.lr)

    if cfg.uniform_init:
        print('uniformly initialize parameters [-%f, +%f]' % (cfg.uniform_init, cfg.uniform_init), file=sys.stderr)
        nn_utils.uniform_init(-cfg.uniform_init, cfg.uniform_init, model.parameters())
    elif cfg.xavier_init:
        print('use xavier initialization', file=sys.stderr)
        nn_utils.xavier_init(model.parameters())

    # load pre-trained word embedding (optional)
    if cfg.glove_embed_path:
        print('load glove embedding from: %s' % cfg.glove_embed_path, file=sys.stderr)
        glove_embedding = GloveHelper(cfg.glove_embed_path)
        glove_embedding.load_to(model.src_embed, vocab.source)

    print('begin training, %d training examples, %d dev examples' % (len(train_set), len(dev_set)), file=sys.stderr)
    print('vocab: %s' % repr(vocab), file=sys.stderr)

    epoch = train_iter = 0
    report_loss = report_examples = report_sup_att_loss = 0.
    history_dev_scores = []
    num_trial = patience = 0
    while True:
        epoch += 1
        epoch_begin = time.time()

        for batch_examples in train_set.batch_iter(batch_size=cfg.batch_size, shuffle=True):
            batch_examples = [e for e in batch_examples if len(e.tgt_actions) <= cfg.decode_max_time_step]
            train_iter += 1
            optimizer.zero_grad()

            ret_val = model.score(batch_examples)
            loss = -ret_val[0]

            # print(loss.data)
            loss_val = torch.sum(loss).data[0]
            report_loss += loss_val
            report_examples += len(batch_examples)
            loss = torch.mean(loss)

            if cfg.sup_attention:
                att_probs = ret_val[1]
                if att_probs:
                    sup_att_loss = -torch.log(torch.cat(att_probs)).mean()
                    sup_att_loss_val = sup_att_loss.data[0]
                    report_sup_att_loss += sup_att_loss_val

                    loss += sup_att_loss

            loss.backward()

            # clip gradient
            if cfg.clip_grad > 0.:
                grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), cfg.clip_grad)

            optimizer.step()

            if train_iter % cfg.log_every == 0:
                log_str = '[Iter %d] encoder loss=%.5f' % (train_iter, report_loss / report_examples)
                if cfg.sup_attention:
                    log_str += ' supervised attention loss=%.5f' % (report_sup_att_loss / report_examples)
                    report_sup_att_loss = 0.

                print(log_str, file=sys.stderr)
                report_loss = report_examples = 0.

        print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stderr)

        if cfg.save_all_models:
            model_file = cfg.save_to + '.iter%d.bin' % train_iter
            print('save model to [%s]' % model_file, file=sys.stderr)
            model.save(model_file)

        # perform validation
        if cfg.dev_file:
            if epoch % cfg.valid_every_epoch == 0:
                print('[Epoch %d] begin validation' % epoch, file=sys.stderr)
                eval_start = time.time()
                eval_results = evaluation.evaluate(dev_set.examples, model, evaluator, cfg,
                    verbose=True, eval_top_pred_only=cfg.eval_top_pred_only)
                dev_score = eval_results[evaluator.default_metric]

                print('[Epoch %d] evaluate details: %s, dev %s: %.5f (took %ds)' % (
                    epoch, eval_results,
                    evaluator.default_metric,
                    dev_score,
                    time.time() - eval_start), file=sys.stderr)

                is_better = history_dev_scores == [] or dev_score > max(history_dev_scores)
                history_dev_scores.append(dev_score)
        else:
            is_better = True

        if cfg.decay_lr_every_epoch and epoch > cfg.lr_decay_after_epoch:
            lr = optimizer.param_groups[0]['lr'] * cfg.lr_decay
            print('decay learning rate to %f' % lr, file=sys.stderr)

            # set new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if is_better:
            patience = 0
            model_file = cfg.save_to + '.bin'
            print('save the current model ..', file=sys.stderr)
            print('save model to [%s]' % model_file, file=sys.stderr)
            model.save(model_file)
            # also save the optimizers' state
            torch.save(optimizer.state_dict(), cfg.save_to + '.optim.bin')
        elif patience < cfg.patience and epoch >= cfg.lr_decay_after_epoch:
            patience += 1
            print('hit patience %d' % patience, file=sys.stderr)

        if epoch == cfg.max_epoch:
            print('reached max epoch, stop!', file=sys.stderr)
            exit(0)

        if patience >= cfg.patience and epoch >= cfg.lr_decay_after_epoch:
            num_trial += 1
            print('hit #%d trial' % num_trial, file=sys.stderr)
            if num_trial == cfg.max_num_trial:
                print('early stop!', file=sys.stderr)
                exit(0)

            # decay lr, and restore from previously best checkpoint
            lr = optimizer.param_groups[0]['lr'] * cfg.lr_decay
            print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

            # load model
            params = torch.load(cfg.save_to + '.bin', map_location=lambda storage, loc: storage)
            model.load_state_dict(params['state_dict'])
            if cfg.cuda:
                model = model.cuda()

            # load optimizers
            if cfg.reset_optimizer:
                print('reset optimizer', file=sys.stderr)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            else:
                print('restore parameters of the optimizers', file=sys.stderr)
                optimizer.load_state_dict(torch.load(cfg.save_to + '.optim.bin'))

            # set new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # reset patience
            patience = 0

    # final setup
    epilogue(cfg, summary_writer)


if __name__ == '__main__':
    train(cfg=init_cfg("./configs/conala.yaml", do_seed=True))
