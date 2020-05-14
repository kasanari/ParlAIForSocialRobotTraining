
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

try:
    from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel, AutoModel
except ImportError:
    raise ImportError('Please run `pip install transformers`.')

from transformers.modeling_utils import SequenceSummary

import torch


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

        self.config.summary_first_dropout = 0.1
        self.config.summary_proj_to_labels = True
        self.config.summary_type = "cls_index"
        self.config.summary_use_proj = True

        if opt["next_sentence_prediction"]:
            self.next_sentence_prediction = True
            self.config.summary_activation = None
            self.config.num_labels = 1
            self.mc_head = SequenceSummary(self.config)  # Multiple choice head
        else:
            self.next_sentence_prediction = False

        if opt["emotion_prediction"]:

            if opt['classes_from_file'] is not None:
                with open(opt['classes_from_file']) as f:
                    self.class_list = f.read().splitlines()
            self.config.summary_activation = "tanh"
            self.emotion_prediction = True
            self.config.num_labels = len(self.class_list) #opt["emotion_prediction"]
            self.emo_head = SequenceSummary(self.config)  # Emotion prediction head
        else:
            self.emotion_prediction = False

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
            total_length = max(self.text_lengths) - 1
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

    def predict(self, head, hidden_states, token_ids):
        logits = head(hidden_states, token_ids).squeeze(-1)
        return logits

    def decode_forced_and_predict(self, context, cands, token_ids):

        model_input = context.repeat(cands.shape[1], 1).unsqueeze(0) # Make one copy of context for each choice

        model_input = torch.cat([model_input, cands], -1) # Add label and distractor to context

        attention_mask = model_input != self.NULL_IDX
        latent, _ = self.transformer(input_ids=model_input, attention_mask=attention_mask)

        true_sentence = latent[:, self.label_inds.item(), ...]

        lm_logits = self.output(true_sentence, label_lengths=self.label_lengths)
        mc_logits = self.predict(self.mc_head, latent, token_ids)

        _, lm_preds = lm_logits.max(dim=2)
        _, mc_preds = mc_logits.max(dim=1)

        if self.emotion_prediction:

            ec_logits = self.predict(self.emo_head, true_sentence, token_ids[:, self.label_inds.item()])
            _, ec_preds = ec_logits.max(dim=1)
 
            return lm_logits, lm_preds, mc_logits, mc_preds, ec_logits, ec_preds
        else:
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
            return self.decode_forced_and_predict(context, cands, mc_token_ids)
        else:
            # use teacher forcing
            scores, preds = self.decode_forced(context, ys)

        return scores, preds, xs
