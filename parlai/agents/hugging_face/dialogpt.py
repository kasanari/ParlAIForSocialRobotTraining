#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from parlai.core.torch_generator_agent import TorchGeneratorAgent, TorchGeneratorModel, PPLMetric
from parlai.core.torch_agent import Output, Batch
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

        if opt["next_sentence_prediction"]:
            self.next_sentence_prediction = True
            self.config.num_labels = 1
            self.mc_head = SequenceSummary(self.config)  # Multiple choice head
            self.mc_labels = torch.LongTensor([0]).to("cuda") # Correct label is always at index 0

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

    def output(self, tensor, label_lengths = None):
        """
        Compute output logits.

        Because we concatenate the context with the labels using the
        `concat_without_padding` function, we must truncate the input tensor to return
        only the scores for the label tokens.
        """

        if label_lengths is not None:
            new_tensors = []
            total_length = max(self.text_lengths)
            label = tensor[:, total_length:(total_length+label_lengths), :]

            new_tensors.append(label)
            tensor = torch.cat(new_tensors, 0)
            return self.lm_head(tensor)


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
    
    def predict(self, hidden_states, token_ids):
        #token_ids = token_ids.unsqueeze(0)
        mc_logits = self.mc_head(hidden_states, token_ids).squeeze(-1)
        return mc_logits

    def decode_forced_and_predict(self, context, cands, token_ids):

        model_input = context.repeat(cands.shape[1], 1).unsqueeze(0) # Make one copy of context for each choice

        model_input = torch.cat([model_input, cands], -1) # Add label and distractor to context

        attention_mask = model_input != self.NULL_IDX
        latent, _ = self.transformer(input_ids=model_input, attention_mask=attention_mask)

        lm_logits = self.output(latent[:, self.label_inds.item(), ...], label_lengths=self.label_lengths)
        mc_logits = self.predict(latent, token_ids)
 
        _, lm_preds = lm_logits.max(dim=2)
        _, mc_preds = mc_logits.max(dim=1)

        return lm_logits, lm_preds, mc_logits, mc_preds

    def forward(self, *xs, ys=None, prev_enc=None, maxlen=None, bsz=None):
        if len(xs) > 2:
            context, text_lengths, cands, mc_token_ids = xs
        else:
            context, text_lengths = xs

        if ys is not None:
            self.text_lengths = text_lengths
            if type(ys) is tuple:
                self.label_lengths = ys[0].shape[1]
                self.label_inds = ys[1]
            else:
                self.label_lengths = ys.shape[1]
        else:
            self.text_lengths = None

        assert ys is not None, "Greedy decoding in TGModel.forward no longer supported."

        if self.next_sentence_prediction:
            lm_logits, lm_preds, mc_logits, mc_preds = self.decode_forced_and_predict(context, cands, mc_token_ids)
            return lm_logits, lm_preds, mc_logits, mc_preds
        else:
            # use teacher forcing
            scores, preds = self.decode_forced(context, ys)

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
        agent.add_argument(
            '--next-sentence-prediction',
            type='bool',
            default=False,
            help='Add next sentence prediction training objective.',
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

    def compute_loss(self, batch, return_output=False):
        """
        Compute and return the loss for the given batch.

        Easily overridable for customized loss functions.

        If return_output is True, the full output from the call to self.model()
        is also returned, via a (loss, model_output) pair.
        """
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')

        if batch.candidate_vecs is not None:
            cands, label_inds, mc_token_ids = self.build_candidates(batch)
            model_output = self.model(batch.text_vec, batch.text_lengths, cands, mc_token_ids, ys=(batch.label_vec, label_inds))

            lm_logits, lm_preds, mc_logits, mc_preds = model_output

            mc_loss = self.criterion(mc_logits.view(-1, mc_logits.size(-1)), label_inds.view(-1))
            self.record_local_metric('mc_loss', AverageMetric.many(mc_loss))
            if batch.candidates is not None:
                candidate = [batch.candidates[0][mc_preds.item()]]

        else:
            model_output = self.model(*self._model_input(batch), ys=batch.label_vec)
            lm_logits, lm_preds, *_ = model_output

        score_view = lm_logits.view(-1, lm_logits.size(-1))
        lm_loss = self.criterion(score_view, batch.label_vec[0].view(-1))
        lm_loss = lm_loss.view(lm_logits.shape[:-1]).sum(dim=1)
        # save loss to metrics
        notnull = batch.label_vec.ne(self.NULL_IDX)
        target_tokens = notnull.long().sum(dim=-1)
        correct = ((batch.label_vec == lm_preds) * notnull).sum(dim=-1)

        self.record_local_metric('lm_loss', AverageMetric.many(lm_loss, target_tokens))
        self.record_local_metric('ppl', PPLMetric.many(lm_loss, target_tokens))
        self.record_local_metric(
            'token_acc', AverageMetric.many(correct, target_tokens)
        )

        lm_coef = 1.0
        mc_coef = 1.0

        if mc_loss is None:
            loss = lm_loss
        else:
            loss = lm_loss * lm_coef + mc_loss * mc_coef # Combined loss
            self.record_local_metric('total_loss', AverageMetric.many(loss))

        if return_output:
            return (loss, model_output + tuple(candidate))
        else:
            return loss

    def build_candidates(self, batch):

        label_id = None

        if batch.candidates == None:
            label_id = 0
        else:
            for i, candidate in enumerate(batch.candidates[0]):
                if candidate == batch.labels[0]:
                    label_id = i
                    break
        
        distractor_id = 1 - label_id

        distractor_length = [batch.candidate_vecs[0][distractor_id].size(0)] #TODO make this more general

        label_token_ids = [x+y-1 for x, y in zip(batch.text_lengths, batch.label_lengths)]
        distractor_token_ids = [x+y-1 for x, y in zip(batch.text_lengths, distractor_length)]
        mc_token_ids = torch.LongTensor([label_token_ids, distractor_token_ids]).to("cuda")
        label_inds = torch.LongTensor([label_id]).to("cuda") # Correct label is always at index 0 #TODO make this not the case #TODO make work for bigger batch size
        cands = padded_3d(batch.candidate_vecs, use_cuda=True, pad_idx=self.NULL_IDX)

        return cands, label_inds, torch.t(mc_token_ids)

    def vectorize(self, *args, **kwargs):
        if self.opt["next_sentence_prediction"]: # Hacky solution to include candidate vecs in batch
            self.rank_candidates = True
        obs = super().vectorize(*args, **kwargs)
        if self.opt["next_sentence_prediction"]:
            self.rank_candidates = False
        return obs

    def _dummy_batch(self, batchsize, maxlen):
        """
        Create a dummy batch.

        This is used to preinitialize the cuda buffer, or otherwise force a
        null backward pass after an OOM.

        If your model uses additional inputs beyond text_vec and label_vec,
        you will need to override it to add additional fields.
        """
        return Batch(
            text_vec=torch.ones(batchsize, maxlen).long().cuda(),
            label_vec=torch.ones(batchsize, maxlen).long().cuda(),
            candidate_vecs=[[torch.ones(maxlen).long().cuda() for _ in range(2)] for _ in range(batchsize)],
            text_lengths=[maxlen] * batchsize,
            label_lengths=[maxlen] * batchsize,
        )

    def train_step(self, batch):
        """
        Train on a single batch of examples.
        """
        # helps with memory usage
        # note we want to use the opt's batchsize instead of the observed batch size
        # in case dynamic batching is in use
        self._init_cuda_buffer(self.opt['batchsize'], self.label_truncate or 256)
        self.model.train()
        self.zero_grad()

        try:
            loss, model_output = self.compute_loss(batch, return_output=True)
            candidate = model_output[-1]
            self.backward(loss)
            self.update_params()
            return Output([candidate])
        except RuntimeError as e:
            # catch out of memory exceptions during fwd/bck (skip batch)
            if 'out of memory' in str(e):
                print(
                    '| WARNING: ran out of memory, skipping batch. '
                    'if this happens frequently, decrease batchsize or '
                    'truncate the inputs to the model.'
                )
                self.global_metrics.add('skipped_batches', SumMetric(1))
                # gradients are synced on backward, now this model is going to be
                # out of sync! catch up with the other workers
                self._init_cuda_buffer(8, 8, True)
            else:
                raise e