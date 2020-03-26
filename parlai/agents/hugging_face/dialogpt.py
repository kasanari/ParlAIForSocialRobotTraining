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

try:
    from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel, AutoModel
except ImportError:
    raise ImportError('Please run `pip install transformers`.')

from transformers.modeling_utils import SequenceSummary

import torch


class Batch(AttrDict):
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
        distractor_vec=None,
        distractors=None,
        distractor_lengths=None,
        ** kwargs,
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
            distractor_vec=distractor_vec,
            distractors=distractors,
            distractor_lengths=distractor_lengths,
            **kwargs,
        )

############################################
# Modules
############################################


class GPT2Decoder(torch.nn.Module):
    """
    GPT2 Decoder.

    This decoder is initialized with the pretrained model from Hugging Face.
    """

    def __init__(self, opt, dict):
        super().__init__()
        # load model
        model_sz = opt['gpt2_size']
        self.transformer = AutoModel.from_pretrained(
            f"microsoft/DialoGPT-{model_sz}")
        # add special tokens
        self.start_idx = dict.start_idx
        self.null_idx = dict.null_idx
        self.add_start_token = False
        if opt['add_special_tokens']:
            self.transformer.resize_token_embeddings(len(dict.tokenizer))
            self.add_start_token = opt['add_start_token']
        # use cuda
        self.use_cuda = not opt['no_cuda'] and torch.cuda.is_available()

    def forward(self, input, encoder_states, incr_state=None):
        if incr_state is None:
            # first step
            if (
                not self.add_start_token
                and input.size(1) == 1
                and int(input[0][0]) == self.start_idx
            ):
                # generating: ignore the start token
                model_input = encoder_states
            else:
                # forced decoding: concatenate the context
                # with the labels
                model_input, _ = concat_without_padding(
                    encoder_states,
                    input,
                    use_cuda=self.use_cuda,
                    null_idx=self.null_idx,
                )
        else:
            # generation: get the last token input
            model_input = input[:, -1].unsqueeze(1)

        attention_mask = model_input != self.null_idx

        #position_ids = torch.LongTensor(list(range(model_input.size(1)))).repeat(model_input.size(0), 1).to("cuda")

        transformer_outputs = self.transformer(
            model_input, past=incr_state, attention_mask=attention_mask
        )
        hidden_states = transformer_outputs[0]
        new_incr_state = transformer_outputs[1]

        return hidden_states, new_incr_state


class HFGPT2Model(TorchGeneratorModel):
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
        self.encoder = IdentityLayer()
        self.decoder = GPT2Decoder(opt, dict)
        self.config = self.decoder.transformer.config
        self.lm_head = torch.nn.Linear(
            self.config.n_embd, self.config.vocab_size, bias=False
        )

        if opt["next_sentence_prediction"]:
            self.config.num_labels = 1
            self.mc_head = SequenceSummary(self.config)  # Multiple choice head

        self._tie_weights(self.lm_head, self.decoder.transformer.wte)
        # add start token
        self.add_start_token = opt['add_special_tokens'] and opt['add_start_token']
        # used to reverse concatenation of context and labels
        self.text_lengths = None

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

    def decode_forced(self, encoder_states, ys):
        """
        Override to get rid of start token input.
        """
        if self.add_start_token:
            return super().decode_forced(encoder_states, ys)
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        latent, _ = self.decoder(inputs, encoder_states)
        logits = self.output(latent)
        _, preds = logits.max(dim=2)
        return logits, preds

    def forward(self, *xs, ys=None, prev_enc=None, maxlen=None, bsz=None):
        model_input, text_lengths = xs
        if ys is not None:
            self.text_lengths = text_lengths
        else:
            self.text_lengths = None

        return super().forward(
            model_input, ys=ys, prev_enc=prev_enc, maxlen=maxlen, bsz=bsz
        )


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
        if opt["next_sentence_prediction"] and opt['batchsize'] > 1:
                raise RuntimeError("Next sentence prediction is not implemented for batchsize > 1")
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
        return HFGPT2Model(self.opt, self.dict)

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

    def compute_loss(self, batch, return_output=False):
        """
        Compute and return the loss for the given batch.

        Easily overridable for customized loss functions.

        If return_output is True, the full output from the call to self.model()
        is also returned, via a (loss, model_output) pair.
        """
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')
        model_output = self.model(*self._model_input(batch), ys=batch.label_vec)
        scores, preds, encoder_state = model_output
        score_view = scores.view(-1, scores.size(-1))
        lm_loss = self.criterion(score_view, batch.label_vec.view(-1))
        lm_loss = lm_loss.view(scores.shape[:-1]).sum(dim=1)

        mc_loss = None

        bsz = batch.text_vec.shape[0]
        if hasattr(self.model, 'mc_head'):
            context = batch.text_vec

            if batch.candidate_vecs is not None:
                choices = padded_3d(batch.candidate_vecs,
                                    use_cuda=True, pad_idx=self.NULL_IDX)

                model_input = encoder_state.repeat(2, 1).unsqueeze(0) # Make two copies of context

                model_input = torch.cat([model_input, choices], -1) # Add label and distractor
                attention_mask = model_input != self.NULL_IDX

                hidden_states, _ = self.model.decoder.transformer(
                    input_ids=model_input, attention_mask=attention_mask)

                mc_labels = torch.LongTensor([0]).to("cuda") # Correct label is always at index 0
                if bsz > 1:
                    mc_labels = mc_labels.repeat(bsz, 1)
                mc_token_ids = torch.LongTensor([batch.text_lengths[0] + batch.label_lengths[0] - 1,
                                                batch.text_lengths[0] + batch.distractor_lengths[0] - 1]).to("cuda")
                mc_token_ids = mc_token_ids.unsqueeze(0)
                mc_logits = self.model.mc_head(
                    hidden_states, mc_token_ids).squeeze(-1)

                if batch.label_vec is not None:
                    mc_loss = self.criterion(
                        mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))

                _, mc_preds = mc_logits.max(dim=1)
                mc_correct = ((mc_labels == mc_preds)).sum().unsqueeze(-1)

                #self.record_local_metric('mc_accuracy', AverageMetric.many(mc_correct))
                self.record_local_metric('mc_loss', AverageMetric.many(mc_loss))

                candidate = [batch.candidates[0][mc_preds.item()]]


        # save loss to metrics
        notnull = batch.label_vec.ne(self.NULL_IDX)
        target_tokens = notnull.long().sum(dim=-1)
        correct = ((batch.label_vec == preds) * notnull).sum(dim=-1)

        self.record_local_metric(
            'lm_loss', AverageMetric.many(lm_loss, target_tokens))

        self.record_local_metric('ppl', PPLMetric.many(lm_loss, target_tokens))
        self.record_local_metric(
            'token_acc', AverageMetric.many(correct, target_tokens)
        )
        # actually do backwards loss
        lm_loss = lm_loss.sum()
        lm_loss /= target_tokens.sum()  # average loss per token

        lm_coef = 0.5
        mc_coef = 0.5

        if mc_loss is None:
            loss = lm_loss
        else:
            loss = lm_loss * lm_coef + mc_loss * mc_coef # Combined loss

        if return_output:
            return (loss, model_output, candidate)
        else:
            return loss

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

    # def batchify(self, obs_batch, sort=False):

    def vectorize(self, *args, **kwargs):

        obs = super().vectorize(*args, **kwargs)

        distractor = obs["distractor_label"]
        obs["distractor_vec"] = self._vectorize_text(
            distractor[0], add_start=False, add_end=True, truncate=kwargs["label_truncate"], truncate_left=False)

        return obs

    def batchify(self, obs_batch, sort=False):
        batch: Batch = super().batchify(obs_batch)

        exs = obs_batch
        label_vecs = [ex.get("distractor_vec", self.EMPTY) for ex in exs]
        labels = [ex.get("distractor_label") for ex in exs]

        y_lens = [y.shape[0] for y in label_vecs]

        ys, y_lens = self._pad_tensor(label_vecs)

        batch["distractor_vec"] = ys
        batch["distractors"] = labels
        batch["distractor_lengths"] = y_lens

        xs = [ex.get("text_vec", self.EMPTY) for ex in exs]
        x_lens = [x.shape[0] for x in xs]

        batch["mc_token_ids"] = [
            (len_x + len_y) - 1 for len_x, len_y in zip(x_lens, y_lens)]

        return batch

    # def _dummy_batch(self, batchsize, maxlen):
    #     """
    #     Create a dummy batch.

    #     This is used to preinitialize the cuda buffer, or otherwise force a
    #     null backward pass after an OOM.

    #     If your model uses additional inputs beyond text_vec and label_vec,
    #     you will need to override it to add additional fields.
    #     """
    #     return Batch(
    #         text_vec=torch.ones(batchsize, maxlen).long().cuda(),
    #         label_vec=torch.ones(batchsize, 2).long().cuda(),
    #         distractor_vec=torch.ones(batchsize, 2).long().cuda(),
    #         text_lengths=[maxlen] * batchsize,
    #         mc_token_ids=None
    #     )

    def _set_label_cands_vec(self, obs, add_start, add_end, truncate):
        """
        Set the 'label_candidates_vec' field in the observation.

        Useful to override to change vectorization behavior
        """
        if 'label_candidates_vecs' in obs:
            if truncate is not None:
                # check truncation of pre-computed vectors
                vecs = obs['label_candidates_vecs']
                for i, c in enumerate(vecs):
                    vecs[i] = self._check_truncate(c, truncate)
        elif obs.get('label_candidates'):
            obs.force_set('label_candidates', list(obs['label_candidates']))
            obs['label_candidates_vecs'] = [
                self._vectorize_text(c, add_start, add_end, truncate, False)
                for c in obs['label_candidates']
            ]
        return obs    

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
            loss, _, candidates = self.compute_loss(batch, return_output=True)
            self.backward(loss)
            self.update_params()
            return Output(text=candidates)
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

    def eval_step(self, batch):
        """
        Evaluate a single batch of examples.
        """
        if batch.text_vec is None and batch.image is None:
            return
        if batch.text_vec is not None:
            bsz = batch.text_vec.size(0)
        else:
            bsz = len(batch.image)
        self.model.eval()
        cand_scores = None
        token_losses = None

        if batch.label_vec is not None:
            # calculate loss on targets with teacher forcing
            loss, model_output, candidates = self.compute_loss(batch, return_output=True)
            if self.output_token_losses:
                token_losses = self._construct_token_losses(
                    batch.label_vec, model_output
                )

        preds = None
        if self.skip_generation:
            warn_once(
                "--skip-generation does not produce accurate metrics beyond ppl",
                RuntimeWarning,
            )
        else:
            maxlen = self.label_truncate or 256
            beam_preds_scores, _ = self._generate(batch, self.beam_size, maxlen)
            preds, scores = zip(*beam_preds_scores)

        cand_choices = None
        # TODO: abstract out the scoring here
        if self.rank_candidates:
            # compute roughly ppl to rank candidates
            cand_choices = []
            encoder_states = self.model.encoder(*self._encoder_input(batch))
            for i in range(bsz):
                num_cands = len(batch.candidate_vecs[i])
                enc = self.model.reorder_encoder_states(encoder_states, [i] * num_cands)
                cands, _ = self._pad_tensor(batch.candidate_vecs[i])
                scores, _ = self.model.decode_forced(enc, cands)
                cand_losses = F.cross_entropy(
                    scores.view(num_cands * cands.size(1), -1),
                    cands.view(-1),
                    reduction='none',
                ).view(num_cands, cands.size(1))
                # now cand_losses is cands x seqlen size, but we still need to
                # check padding and such
                mask = (cands != self.NULL_IDX).float()
                cand_scores = (cand_losses * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
                _, ordering = cand_scores.sort()
                cand_choices.append([batch.candidates[i][o] for o in ordering])

        text = candidates if candidates is not None else None
        if text and self.compute_tokenized_bleu:
            # compute additional bleu scores
            self._compute_fairseq_bleu(batch, preds)
            self._compute_nltk_bleu(batch, text)
        return Output(text, cand_choices, token_losses=token_losses)

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
        encoder_states = model.encoder(*self._encoder_input(batch))
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
            torch.LongTensor([self.START_IDX]).expand(
                bsz * beam_size, 1).to(dev)
        )

        inds = torch.arange(bsz).to(dev).unsqueeze(
            1).repeat(1, beam_size).view(-1)
        encoder_states = model.reorder_encoder_states(encoder_states, inds)
        incr_state = None

        for _ts in range(max_ts):
            if all((b.is_done() for b in beams)):
                    # exit early if possible
                break

            score, incr_state = model.decoder(
                decoder_input, encoder_states, incr_state)
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
            decoder_input = torch.index_select(
                decoder_input, 0, incr_state_inds)
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
        beam_preds_scores = [n_best_list[0]
                             for n_best_list in n_best_beam_preds_scores]

        return beam_preds_scores, beams
