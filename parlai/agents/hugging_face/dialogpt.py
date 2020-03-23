#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.torch_generator_agent import TorchGeneratorAgent, TorchGeneratorModel, PPLMetric
from parlai.agents.hugging_face.dict import DialogptDictionaryAgent
from parlai.agents.hugging_face.dialogger import DialoggerHistory
from parlai.utils.misc import warn_once, AttrDict
from parlai.utils.torch import IdentityLayer, concat_without_padding, padded_tensor
from parlai.core.metrics import SumMetric, AverageMetric, BleuMetric, FairseqBleuMetric

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
                not self.add_start_token and
                input.size(1) == 1 and
                int(input[0][0]) == self.start_idx
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
        loss = self.criterion(score_view, batch.label_vec.view(-1))
        loss = loss.view(scores.shape[:-1]).sum(dim=1)
        # save loss to metrics
        notnull = batch.label_vec.ne(self.NULL_IDX)
        target_tokens = notnull.long().sum(dim=-1)
        correct = ((batch.label_vec == preds) * notnull).sum(dim=-1)

        if hasattr(self, 'mc_head'):
            hidden_states, _ = self.model.decoder.transformer(
                input_ids=encoder_state)

            mc_logits = self.model.mc_head(
                hidden_states, batch.mc_token_ids).squeeze(-1)

            if batch.label_vec is not None:
                mc_loss_fct = CrossEntropyLoss()
                mc_loss = loss_fct(
                    mc_logits.view(-1, mc_logits.size(-1)), batch.label_vec.view(-1))

        self.record_local_metric(
            'loss', AverageMetric.many(loss, target_tokens))
        self.record_local_metric('ppl', PPLMetric.many(loss, target_tokens))
        self.record_local_metric(
            'token_acc', AverageMetric.many(correct, target_tokens)
        )
        # actually do backwards loss
        loss = loss.sum()
        loss /= target_tokens.sum()  # average loss per token
        if return_output:
            return (loss, model_output)
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
        obs["distractor_vec"] = self._vectorize_text(distractor[0], add_start=False, add_end=True, truncate=kwargs["label_truncate"], truncate_left=False)

        return obs
