from parlai.core.torch_generator_agent import TorchGeneratorAgent, TorchGeneratorModel
from parlai.agents.hugging_face.dict import DialogptDictionaryAgent
from transformers import GPT2Config, GPT2LMHeadModel
from parlai.utils.torch import IdentityLayer, concat_without_padding

import torch


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

    return model


class HFGPT2Wrapper(torch.nn.Module):

    def __init__(self, transformer, dictionary):
        super().__init__()

        self.transformer = transformer

        self.dict = dictionary

        # use cuda
        self.use_cuda = torch.cuda.is_available()

    def forward(self, decoder_input, encoder_states, incr_state=None):

        attention_mask = None
            # concatenate the context and the generated tokens
        model_input, _ = concat_without_padding(
                encoder_states,
                decoder_input,
                use_cuda=self.use_cuda,
                # null_idx=self.null_idx,
            )
            # attention_mask = model_input != self.null_idx

        transformer_outputs = self.transformer(
            model_input, past=incr_state, attention_mask=attention_mask)
        hidden_states = transformer_outputs[0]
        new_incr_state = transformer_outputs[1]

        return hidden_states, new_incr_state


class DialoggerModel(TorchGeneratorModel):

    def __init__(self, opt: dict, dictionary: DialogptDictionaryAgent):
        super().__init__(start_idx=dictionary.start_idx, end_idx=dictionary.end_idx)
        # load model
        model_sz = opt['gpt2_size']

        if model_sz == 'medium':
            fle_key = '345'
        elif model_sz == 'large':
            fle_key = '762'
        else:
            fle_key = '117'  # small

        dialogpt = load_model(fle_key, model_sz)

        # init the model
        self.encoder = IdentityLayer()  # Decoder-only model, so encoder is identity layer
        self.decoder = HFGPT2Wrapper(dialogpt.transformer, dictionary)
        self.config = dialogpt.config
        self.lm_head = dialogpt.lm_head

        self._tie_weights(self.lm_head, self.decoder.transformer.wte)

    def output(self, decoder_output):
        return self.lm_head(decoder_output)

    def _tie_weights(self, output_embeddings, input_embeddings):
        """
        Make sure the input and output word embeddings share the same weights
        """
        output_embeddings.weight.data=input_embeddings.weight.data

    def reorder_encoder_states(self, encoder_states, indices):
        enc=torch.index_select(encoder_states, 0, indices)
        return enc

    def reorder_decoder_incremental_state(self, incremental_state, inds):
        new_incr_state=[]
        for layer_past in incremental_state:
            new_incr_state.append(torch.index_select(layer_past, 1, inds))

        return tuple(new_incr_state)


class DialoggerAgent(TorchGeneratorAgent):

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

    def build_model(self):
        return DialoggerModel(self.opt, self.dict)

    def _encoder_input(self, batch):
        return (batch.text_vec,)

    @staticmethod
    def dictionary_class():
        return DialogptDictionaryAgent

    def _set_text_vec(self, *args, **kwargs):
        """
        Add the start and end token to the text.
        """
        obs=super()._set_text_vec(*args, **kwargs)

        # if 'text_vec' in obs:
        #     obs.force_set(
        #         'text_vec', self._add_start_end_tokens(
        #             obs['text_vec'], True, True)
        #     )
        #     obs['added_start_end'] = True

        return obs
