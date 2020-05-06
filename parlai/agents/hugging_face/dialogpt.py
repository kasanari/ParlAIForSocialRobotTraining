#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from parlai.core.torch_generator_agent import TorchGeneratorAgent, TorchGeneratorModel, PPLMetric
from parlai.core.torch_agent import Output, Batch, History
from parlai.agents.hugging_face.dict import DialogptDictionaryAgent
from parlai.agents.hugging_face.dialogger import DialoggerHistory
from parlai.utils.misc import warn_once, AttrDict
from parlai.utils.torch import IdentityLayer, concat_without_padding, padded_tensor, padded_3d
from parlai.core.metrics import SumMetric, AverageMetric, BleuMetric, FairseqBleuMetric, ExactMatchMetric
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from collections import deque

from .dialogpt_model import DialoGPTModel

try:
    from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel, AutoModel
except ImportError:
    raise ImportError('Please run `pip install transformers`.')

from transformers.modeling_utils import SequenceSummary

import torch

############################################
# Modules
############################################


class Batch(AttrDict):
    """
    Batch is a namedtuple containing data being sent to an agent.

    This is the input type of the train_step and eval_step functions.
    Agents can override the batchify function to return an extended namedtuple
    with additional fields if they would like, though we recommend calling the
    parent function to set up these fields as a base.

    :param text_vec:
        bsz x seqlen tensor containing the parsed text data.

    :param text_lengths:
        list of length bsz containing the lengths of the text in same order as
        text_vec; necessary for pack_padded_sequence.

    :param label_vec:
        bsz x seqlen tensor containing the parsed label (one per batch row).

    :param label_lengths:
        list of length bsz containing the lengths of the labels in same order as
        label_vec.

    :param labels:
        list of length bsz containing the selected label for each batch row (some
        datasets have multiple labels per input example).

    :param valid_indices:
        list of length bsz containing the original indices of each example in the
        batch. we use these to map predictions back to their proper row, since e.g.
        we may sort examples by their length or some examples may be invalid.

    :param candidates:
        list of lists of text. outer list has size bsz, inner lists vary in size
        based on the number of candidates for each row in the batch.

    :param candidate_vecs:
        list of lists of tensors. outer list has size bsz, inner lists vary in size
        based on the number of candidates for each row in the batch.

    :param image:
        list of image features in the format specified by the --image-mode arg.

    :param observations:
        the original observations in the batched order
    """

    def __init__(
        self,
        text_vec=None,
        text_lengths=None,
        label_vec=None,
        label_lengths=None,
        labels=None,
        valid_indices=None,
        candidates=None,
        candidate_vecs=None,
        image=None,
        observations=None,
        emotion=None,
        **kwargs,
    ):
        super().__init__(
            text_vec=text_vec,
            text_lengths=text_lengths,
            label_vec=label_vec,
            label_lengths=label_lengths,
            labels=labels,
            valid_indices=valid_indices,
            candidates=candidates,
            candidate_vecs=candidate_vecs,
            image=image,
            observations=observations,
            emotion=emotion,
            **kwargs,
        )


class DialoGPTHistory(History):
    """
    Handles tokenization history.
    """

    def get_history_vec(self):

        history = deque(maxlen=self.max_len)
        for vec in self.history_vecs:
            history.extend(vec)
            history.extend([self.dict.end_idx])

        return history


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
        agent.add_argument(
            '--emotion-prediction',
            type='bool',
            default=False,
            help='Add emotion prediction training objective.',
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

        #self.delimiter = opt.get('delimiter', '\n')
        self.delimiter_tok = [self.dict[self.dict.end_token]]
        self._global_end_token = self.dict[self.dict.end_token]
        self.use_cuda = not opt['no_cuda'] and torch.cuda.is_available()

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
        return DialoGPTHistory

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

            score, incr_state = model.transformer(input_ids=model_input, past=incr_state, attention_mask=attention_mask)

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
        emotions = ["surprised", "excited", "angry", "proud", "sad", "annoyed", "grateful", 
                "lonely", "afraid", "terrified", "guilty", "impressed", "disgusted", "hopeful", 
                "confident", "furious", "anxious", "anticipating", "joyful", "nostalgic", 
                "disappointed", "prepared", "jealous", "content", "devastated", "embarrassed", 
                "caring", "sentimental", "trusting", "ashamed", "apprehensive", "faithful"]
        
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')

        if batch.candidate_vecs is not None:
            cands, label_inds, mc_token_ids = self.build_candidates(batch)
            model_output = self.model(batch.text_vec, batch.text_lengths, cands, mc_token_ids, ys=(batch.label_vec, label_inds))

            if self.opt["emotion_prediction"]:
                lm_logits, lm_preds, mc_logits, mc_preds, ec_logits, ec_preds = model_output
            else:
                lm_logits, lm_preds, mc_logits, mc_preds = model_output

            if (batch.emotion is not None) and self.opt["emotion_prediction"]:
                emo_index = torch.tensor(emotions.index(batch.emotion[0])).unsqueeze(0)

                if self.use_cuda:
                    emo_index = emo_index.cuda()

                ec_loss = self.criterion(ec_logits.view(-1, ec_logits.size(-1)), emo_index.view(-1))
                self.record_local_metric("ec_loss", AverageMetric.many(ec_loss))
                predicted_emotion = emotions[ec_preds.item()]
                self.metrics['ec_accuracy'] = self.metrics.get('ec_accuracy') + ExactMatchMetric.compute(predicted_emotion, batch.emotion)
                self.record_local_metric('ec_accuracy', [self.metrics['ec_accuracy']])
            else:
                ec_loss = None

            mc_loss = self.criterion(mc_logits.view(-1, mc_logits.size(-1)), label_inds.view(-1))
            self.record_local_metric('mc_loss', AverageMetric.many(mc_loss))

            if batch.candidates is not None:
                candidate = batch.candidates[0][mc_preds.item()]
                self.metrics['mc_accuracy'] = self.metrics.get('mc_accuracy') + ExactMatchMetric.compute(candidate, batch.labels)
                self.record_local_metric('mc_accuracy', [self.metrics['mc_accuracy']])

        else:
            model_output = self.model(*self._model_input(batch), ys=batch.label_vec)
            lm_logits, lm_preds, *_ = model_output
            mc_loss = None
            ec_loss = None

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
        ec_coef = 1.0

        if (mc_loss is None) and (ec_loss is None): 
            total_loss = None
        else:
            total_loss = lm_loss
            if mc_loss is not None:
                total_loss += (mc_loss * mc_coef)
            if ec_loss is not None:
                total_loss += (ec_loss * ec_coef)

        if total_loss is not None:
            loss = total_loss
            self.record_local_metric('total_loss', AverageMetric.many(total_loss))
        else:
            loss = lm_loss

        if return_output:
            return (loss, model_output)
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

        mc_token_ids = torch.tensor([label_token_ids, distractor_token_ids])
        label_inds = torch.tensor([label_id]) # Correct label is always at index 0 #TODO make this not the case #TODO make work for bigger batch size

        if self.use_cuda:
            mc_token_ids = mc_token_ids.cuda()
            label_inds = label_inds.cuda()

        cands = padded_3d(batch.candidate_vecs, use_cuda=self.use_cuda, pad_idx=self.NULL_IDX)

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
    
    def batchify(self, obs_batch, sort=False):
        """
        Override so that we can add memories to the Batch object.
        """
        batch = super().batchify(obs_batch, sort)
        batch.emotion = [ex["emotion"] for ex in obs_batch]
        return batch

    def _set_text_vec(self, obs, history, truncate):
        """
        Set the 'text_vec' field in the observation.

        Useful to override to change vectorization behavior
        """

        if 'text' not in obs:
            return obs

        if 'text_vec' not in obs:
            # text vec is not precomputed, so we set it using the history
            history_string = history.get_history_str()
            # when text not exist, we get text_vec from history string
            # history could be none if it is an image task and 'text'
            # filed is be empty. We don't want this
            if history_string is None:
                return obs
            obs['full_text'] = history_string
            if history_string:
                history_vec = history.get_history_vec()

            situation = self.dict.txt2vec(obs['situation']) + [self.dict.end_idx]
            obs['text_vec'] = situation + list(history_vec)

        # check truncation
        if obs.get('text_vec') is not None:
            truncated_vec = self._check_truncate(obs['text_vec'], truncate, True)
            obs.force_set('text_vec', torch.LongTensor(truncated_vec))
        return obs


