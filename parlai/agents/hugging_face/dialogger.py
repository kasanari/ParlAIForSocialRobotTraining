from parlai.core.torch_agent import TorchAgent, Batch, Output, History
from parlai.core.opt import Opt
from parlai.agents.hugging_face.dict import DialogptDictionaryAgent
from parlai.utils.torch import IdentityLayer, concat_without_padding

import torch
import torch.nn.functional as F
from torch import Tensor

from transformers import GPT2Config, GPT2LMHeadModel, AutoModel

from collections import deque

def load_model(fle_key: str, model_sz: str) -> GPT2LMHeadModel:

    # model = AutoModel.from_pretrained(f"microsoft/DialoGPT-{model_sz}")

    weights = torch.load(
        f'parlai/agents/hugging_face/dialogpt/{fle_key}M/{model_sz}_ft.pkl')
    cfg = GPT2Config.from_json_file(
        f'parlai/agents/hugging_face/dialogpt/{fle_key}M/config.json')

    #fix misused key value
    weights["lm_head.weight"] = weights["lm_head.decoder.weight"]
    weights.pop("lm_head.decoder.weight", None)

    model = GPT2LMHeadModel(cfg)

    model.load_state_dict(weights)
    model.to('cuda')
    return model

class DialoggerHistory(History):
    """
    Handles tokenization history.
    """

    def get_history_vec(self):

        if len(self.history_vecs) < 5: # Limit history to two turns 
            turns = self.history_vecs
        else:
            turns = self.history_vecs[-5:]

        history = deque(maxlen=self.max_len)
        for vec in turns:
            history.extend(vec)
            history.extend([self.dict.end_idx])

        return history

class DialoggerAgent(TorchAgent):

    def __init__(self, opt : Opt):
        super().__init__(opt)

        self.model : GPT2LMHeadModel = self.build_model()
        self.dict : DialogptDictionaryAgent = self.build_dictionary()

    @classmethod
    def add_cmdline_args(cls, argparser):
        agent=argparser.add_argument_group('Gpt2 Args')
        agent.add_argument(
            '--gpt2-size',
            type=str,
            default='small',
            choices=['small', 'medium', 'large'],
            help='Which size model to initialize.',
        )
        super(DialoggerAgent, cls).add_cmdline_args(argparser)
        return agent

    def vectorize(self, *args, **kwargs):
        kwargs['add_start'] = True
        kwargs['add_end'] = True
        return super().vectorize(*args, **kwargs)

    def build_model(self) -> GPT2LMHeadModel:
        model_sz = self.opt['gpt2_size']

        if model_sz == 'medium':
            fle_key = '345'
        elif model_sz == 'large':
            fle_key = '762'
        else:
            fle_key = '117'  # default to small

        return load_model(fle_key, model_sz) 

    @staticmethod
    def dictionary_class():
        return DialogptDictionaryAgent

    @staticmethod
    def history_class():
        return DialoggerHistory

    def generate(self, batch_size, context) -> Tensor:

        predictions : Tensor
        past : Tensor
        response : Tensor = torch.tensor([[]], dtype=torch.long, device="cuda")
        next_token = context[:, -1:] # Start decoding with eos token

        _, past = self.model(context[:, :-1]) # Feed context to model and calculate embedding

        while True:

            predictions, past = self.model(next_token, past=past) # get predictions and incremental state from language model
      
            logits = predictions[0, -1, :].float()

            filtered_logits = top_p_filtering(logits) # filter logits with top p filtering
            probabilities = F.softmax(filtered_logits, dim=-1) # normalize
            next_token = torch.multinomial(probabilities, 1) # sample from distribution
            next_token = torch.unsqueeze(next_token, -1)

            response = torch.cat([response, next_token], dim=-1) # add sampled token to list of generated tokens

            if next_token.item() == self.END_IDX:
                return response[:, :-1]


    def train_step(self, batch : Batch) -> float:

        outputs = self.model(batch.text_vec, labels=batch.labels)
        loss, logits = outputs[:2]

        return loss


    def eval_step(self, batch: Batch) -> Output:
        
        self.model.eval()

        if batch.text_vec is not None:
            bsz = batch.text_vec.size(0)

        tokens = self.generate(bsz, batch.text_vec)

        text = [self._v2t(t) for t in tokens]

        return Output(text)

    def _v2t(self, vec):
        """
        Convert token indices to string of tokens.
        """
        new_vec = []
        if hasattr(vec, 'cpu'):
            vec = vec.cpu()
        for i in vec:
            new_vec.append(i)
        return self.dict.vec2txt(new_vec)

def top_p_filtering(logits : list, top_p : float = 0.9, filter_value : float =-float('Inf')):
  """
  Credit: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
  """
  assert logits.dim() == 1  # batch size 1 for single word generation
  sorted_logits, sorted_indices = torch.sort(logits, descending=True)
  cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
  # remove tokens with cumulative probability above the threshold
  sorted_indices_to_remove = cumulative_probs > top_p
  # shift the indices to the right to keep also the first token above the threshold
  sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
  sorted_indices_to_remove[..., 0] = 0
  indices_to_remove = sorted_indices[sorted_indices_to_remove]
  logits[indices_to_remove] = filter_value
  return logits