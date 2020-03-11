from parlai.core.torch_agent import TorchAgent, Batch, Output
from parlai.core.opt import Opt
from parlai.agents.hugging_face.dict import DialogptDictionaryAgent
from parlai.utils.torch import IdentityLayer, concat_without_padding

import torch
import torch.nn.functional as F

from transformers import GPT2Config, GPT2LMHeadModel



def load_model(fle_key: str, model_sz: str) -> GPT2LMHeadModel:
    weights = torch.load(
        f'parlai/agents/hugging_face/dialogpt/{fle_key}M/{model_sz}_ft.pkl')
    cfg = GPT2Config.from_json_file(
        f'parlai/agents/hugging_face/dialogpt/{fle_key}M/config.json')

    # fix misused key value
    weights["lm_head.weight"] = weights["lm_head.decoder.weight"]
    weights.pop("lm_head.decoder.weight", None)

    model = GPT2LMHeadModel(cfg)
    model.load_state_dict(weights)
    model.to('cuda')
    return model

class DialoggerAgent(TorchAgent):

    def __init__(self, opt : Opt):
        super().__init__(opt)

        self.model = self.build_model()

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

    def generate(self, batch_size, context) -> torch.Tensor:

        attention_mask = None
        generated_tokens = torch.LongTensor([self.START_IDX]).expand(batch_size, 1).to("cuda")
        past = None

        while True:
            # concatenate the context and the generated tokens
            model_input = torch.cat([context, generated_tokens], dim=-1)

            predictions, past = self.model(context, past=past, attention_mask=attention_mask) # get predictions and incremental state from language model
            logits = predictions[0, -1, :]
            filtered_logits = top_p_filtering(logits) # filter logits with top p filtering
            probabilities = F.softmax(filtered_logits, dim=-1) # normalize
            next_token = torch.multinomial(probabilities, 1) # sample from distribution
            next_token = torch.unsqueeze(next_token, -1)
            generated_tokens = torch.cat([generated_tokens, next_token], dim=-1) # add sampled token to list of generated tokens

            if next_token.item() == self.END_IDX:
                return generated_tokens


    def train_step(self, batch : Batch) -> float:

        outputs = self.model(batch.text_vec, labels=batch.labels)
        loss, logits = outputs[:2]

        return loss


    def eval_step(self, batch: Batch) -> Output:

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