#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from parlai.core.torch_generator_agent import TorchGeneratorAgent, TorchGeneratorModel, PPLMetric
from parlai.core.torch_agent import Output
from parlai.agents.hugging_face.dict import DialogptDictionaryAgent
from parlai.agents.hugging_face.dialogger import DialoggerHistory
from parlai.utils.misc import warn_once, AttrDict
from parlai.utils.torch import IdentityLayer, concat_without_padding, padded_tensor, padded_3d
from parlai.core.metrics import SumMetric, AverageMetric, BleuMetric, FairseqBleuMetric
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

try:
    from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel, AutoModel
except ImportError:
    raise ImportError('Please run `pip install transformers`.')

from transformers.modeling_utils import SequenceSummary

import torch

############################################
# Modules
############################################

class DialoGPTModel(TorchGeneratorModel):
    """
    Hugging Face GPT2 Model.

    GPT2 is a multi-layer decoder-only Transformer. As such, the encoder
    is simply an identity layer. The decoder is initialized with pretrained
    weights from Hugging Face. Read more about this model here
    <https://huggingface.co/transformers/model_doc/gpt2.html>.
    """

    def __init__(self, opt, dict):
        self.null_idx, self.start_idx, self.end_idx = self._get_special_tokens(
            opt, dict
        )
        super().__init__(self.null_idx, self.start_idx, self.end_idx)

        # init the model
        self.transformer: GPT2Model = AutoModel.from_pretrained(
            f"microsoft/DialoGPT-{opt['gpt2_size']}")
        self.config = self.transformer.config
        self.lm_head = torch.nn.Linear(
            self.config.n_embd, self.config.vocab_size, bias=False
        )

        self._tie_weights(self.lm_head, self.transformer.wte)
        # add start token
        self.add_start_token = opt['add_special_tokens'] and opt['add_start_token']
        # used to reverse concatenation of context and labels
        self.text_lengths = None

        if opt['add_special_tokens']:
            self.transformer.resize_token_embeddings(len(dict.tokenizer))
            self.add_start_token = opt['add_start_token']
        # use cuda
        self.use_cuda = not opt['no_cuda'] and torch.cuda.is_available()

    def _tie_weights(self, output_embeddings, input_embeddings):
        output_embeddings.weight = input_embeddings.weight

    def _get_special_tokens(self, opt, dict):
        return dict.null_idx, dict.start_idx, dict.end_idx

    def reorder_encoder_states(self, encoder_states, indices):
        enc = torch.index_select(encoder_states, 0, indices)
        return enc

    def output(self, tensor):
        """
        Compute output logits.

        Because we concatenate the context with the labels using the
        `concat_without_padding` function, we must truncate the input tensor to return
        only the scores for the label tokens.
        """
        # get only scores for labels
        if self.text_lengths is not None:
            total_length = max(self.text_lengths)
            to_select = tensor.size(1) - total_length
            if not self.add_start_token:
                to_select = to_select + 1
            if to_select > 0:
                # select only label scores
                bsz = tensor.size(0)
                new_tensors = []
                for i in range(bsz):
                    start = self.text_lengths[i]
                    if not self.add_start_token:
                        start = start - 1
                    end = start + to_select
                    new_tensors.append(tensor[i: i + 1, start:end, :])
                tensor = torch.cat(new_tensors, 0)

        return self.lm_head(tensor)

    def reorder_decoder_incremental_state(self, incremental_state, inds):
        new_incr_state = []
        for layer_past in incremental_state:
            new_incr_state.append(torch.index_select(layer_past, 1, inds))

        return tuple(new_incr_state)

    def decode_forced(self, xs, ys):
        """
        Override to get rid of start token input.
        """
        if self.add_start_token:
            return super().decode_forced(xs, ys)
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)

        model_input, _ = concat_without_padding(
            xs,
            inputs,
            use_cuda=self.use_cuda,
            null_idx=self.null_idx,
        )

        latent, _ = self.transformer(input_ids=model_input)

        logits = self.output(latent)
        _, preds = logits.max(dim=2)
        return logits, preds

    def forward(self, *xs, ys=None, prev_enc=None, maxlen=None, bsz=None):
        model_input, text_lengths = xs
        if ys is not None:
            self.text_lengths = text_lengths
        else:
            self.text_lengths = None

        assert ys is not None, "Greedy decoding in TGModel.forward no longer supported."

        # use teacher forcing
        scores, preds = self.decode_forced(model_input, ys)

        return scores, preds, xs


############################################
# Agent
############################################


class DialogptAgent(TorchGeneratorAgent):
    """
    Hugging Face GPT2 Agent.

    GPT2 is a multi-layer decoder-only Transformer.
    The decoder is initialized with pretrained weights from Hugging Face.
    Read more about this model here
    <https://huggingface.co/transformers/model_doc/gpt2.html>.

    GPT2 comes in four sizes: small, medium, large, and XL. Use the
    flag `--gpt2-size` to choose the size.

    If you are finetuning the Gpt2 agent as a dialogue agent, be sure
    to run `--add-special-tokens True`. To examine the performance of the
    agent out of the box, run with `--add-special-tokens False`, and make
    sure that the batch size is 1.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        agent = argparser.add_argument_group('Gpt2 Args')
        agent.add_argument(
            '--gpt2-size',
            type=str,
            default='small',
            choices=['small', 'medium', 'large'],
            help='Which size model to initialize.',
        )
        agent.add_argument(
            '--add-special-tokens',
            type='bool',
            default=True,
            help='Add special tokens (like PAD, etc.). If False, '
            'Can only use with batch size 1.',
        )
        agent.add_argument(
            '--add-start-token',
            type='bool',
            default=False,
            help='Add start tokens when finetuning.',
        )
        argparser.set_defaults(
            text_truncate=768,
            label_truncate=256,
            dict_maxexs=0,  # skip building dictionary
        )
        super(DialogptAgent, cls).add_cmdline_args(argparser)
        warn_once('WARNING: this model is in beta and the API is subject to change.')
        return agent

    def __init__(self, opt, shared=None):
        if not opt['add_special_tokens'] and opt['batchsize'] > 1:
            raise RuntimeError(
                'If using batchsize > 1, --add-special-tokens must be True.'
            )
        super().__init__(opt, shared)

    @staticmethod
    def dictionary_class():
        """
        Return the dictionary class that this agent expects to use.

        Can be overriden if a more complex dictionary is required.
        """
        return DialogptDictionaryAgent

    def build_model(self, states=None):
        """
        Build and return model.
        """
        return DialoGPTModel(self.opt, self.dict)

    def _model_input(self, batch):
        """
        Override to pass in text lengths.
        """
        return (batch.text_vec, batch.text_lengths)

    def _encoder_input(self, batch):
        return (batch.text_vec,)

    @staticmethod
    def history_class():
        return DialoggerHistory

    def _pad_tensor(self, items):
        """
        Override to always set fp16friendly to False.
        """
        return padded_tensor(
            items, pad_idx=self.NULL_IDX, use_cuda=self.use_cuda, fp16friendly=False,
        )

    def _generate(self, batch, beam_size, max_ts):
        """
        Generate an output with beam search.

        Depending on the options, this may perform greedy/topk/nucleus generation.

        :param Batch batch:
            Batch structure with input and labels
        :param int beam_size:
            Size of each beam during the search
        :param int max_ts:
            the maximum length of the decoded sequence

        :return:
            tuple (beam_pred_scores, n_best_pred_scores, beams)

            - beam_preds_scores: list of (prediction, score) pairs for each sample in
            Batch
            - n_best_preds_scores: list of n_best list of tuples (prediction, score)
            for each sample from Batch
            - beams :list of Beam instances defined in Beam class, can be used for any
            following postprocessing, e.g. dot logging.
        """
        model = self.model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        encoder_states = batch.text_vec
        if batch.text_vec is not None:
            dev = batch.text_vec.device
        else:
            dev = batch.label_vec.device

        bsz = (
            len(batch.text_lengths)
            if batch.text_lengths is not None
            else len(batch.image)
        )
        if batch.text_vec is not None:
            batchsize = batch.text_vec.size(0)
            beams = [
                self._treesearch_factory(dev).set_context(
                    self._get_context(batch, batch_idx)
                )
                for batch_idx in range(batchsize)
            ]
        else:
            beams = [self._treesearch_factory(dev) for _ in range(bsz)]

        # repeat encoder outputs and decoder inputs
        decoder_input = (
            torch.LongTensor([self.START_IDX]).expand(bsz * beam_size, 1).to(dev)
        )

        inds = torch.arange(bsz).to(dev).unsqueeze(1).repeat(1, beam_size).view(-1)
        encoder_states = model.reorder_encoder_states(encoder_states, inds)
        incr_state = None

        for _ts in range(max_ts):
            if all((b.is_done() for b in beams)):
                    # exit early if possible
                break

            if incr_state is None:
                # first step
                model_input = encoder_states
            else:
                # generation: get the last token input
                model_input = decoder_input[:, -1].unsqueeze(1)

            attention_mask = model_input != self.NULL_IDX

            score, incr_state = model.transformer(input_ids=model_input, past=incr_state)

            # only need the final hidden state to make the word prediction
            score = score[:, -1:, :]
            score = model.output(score)
            # score contains softmax scores for bsz * beam_size samples
            score = score.view(bsz, beam_size, -1)
            if self.temperature != 1.0:
                score.div_(self.temperature)
            # force to fp32 to avoid overflow issues during search calculations
            score = F.log_softmax(score, dim=-1, dtype=torch.float32)
            for i, b in enumerate(beams):
                if not b.is_done():
                    b.advance(score[i])
            incr_state_inds = torch.cat(
                [
                    beam_size * i + b.get_backtrack_from_current_step()
                    for i, b in enumerate(beams)
                ]
            )
            incr_state = model.reorder_decoder_incremental_state(
                incr_state, incr_state_inds
            )
            decoder_input = torch.index_select(decoder_input, 0, incr_state_inds)
            selection = torch.cat(
                [b.get_output_from_current_step() for b in beams]
            ).unsqueeze(-1)
            decoder_input = torch.cat([decoder_input, selection], dim=-1)

        # get all finalized candidates for each sample (and validate them)
        n_best_beam_preds_scores = [b.get_rescored_finished() for b in beams]

        if hasattr(self, '_rerank_beams'):
            n_best_beam_preds_scores = self._rerank_beams(
                batch, n_best_beam_preds_scores
            )

        # get the top prediction for each beam (i.e. minibatch sample)
        beam_preds_scores = [n_best_list[0] for n_best_list in n_best_beam_preds_scores]

        return beam_preds_scores, beams

    def _v2t(self, vec, ignore_end_idx=False):
        """
        Convert token indices to string of tokens.
        """
        new_vec = []
        if hasattr(vec, 'cpu'):
            vec = vec.cpu()
        for i in vec:
            if i == self.END_IDX and not ignore_end_idx:
                break
            elif i != self.START_IDX:
                new_vec.append(i)
        return self.dict.vec2txt(new_vec)