from llama import Tokenizer, Decoder, npsoftmax, npmultinominal2D, warp_temperature, warp_topk
import numpy as np
import os
import pdb
import argparse
from loguru import logger

PROMPT_DICT = {
    "prompt_input":
    ("Below is an instruction that describes a task, paired with an input that provides further context. "
     "Write a response that appropriately completes the request.\n\n"
     "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
     ),
    "prompt_no_input":
    ("Below is an instruction that describes a task. "
     "Write a response that appropriately completes the request.\n\n"
     "### Instruction:\n{instruction}\n\n### Response:"),
}
PROMPT = PROMPT_DICT['prompt_no_input']


class Llama:
    def __init__(self, onnxdir='models', config: dict = {}):
        if not os.path.exists(onnxdir):
            logger.error('{} not exist'.format(onnxdir))

        assert os.path.isdir(onnxdir)

        self.DECODER_COUNT = 32
        # EOS token
        self.FINISH_TOKEN = 2
        self.tokenizer = Tokenizer(os.path.join(onnxdir, 'tokenizer.model'))
        self.init = Decoder(onnxdir, 'decoder-{}.onnx', self.DECODER_COUNT)
        self.past = Decoder(onnxdir, 'decoder-past-{}.onnx',
                            self.DECODER_COUNT)
        self.config = config

        # cache
        self.pastkeys = None
        self.pastvalues = None

    # Modified transformers.models.llama.modeling_llama._make_causal_mask with np.array
    def _make_causal_mask(self,
                          input_ids_shape,
                          dtype,
                          past_key_values_length: int = 0):
        """    
        Make causal mask used for bi-directional self-attention. 
        Output triangle-matrix if `past_key_values_length`=0
        Padding left if `past_key_values_length`>0
        """
        bsz, tgt_len = input_ids_shape
        mask = np.full((tgt_len, tgt_len), fill_value=np.finfo(dtype).min)

        mask_cond = np.arange(mask.shape[1])
        cond = mask_cond < (mask_cond + 1).reshape(-1, 1)
        mask = np.ma.array(mask, mask=cond, fill_value=0).filled()
        # masked_fill_result = np.ma.masked_fill_(mask, condition_row_array)

        if past_key_values_length > 0:
            mask = np.concatenate([
                np.zeros((tgt_len, past_key_values_length), dtype=dtype), mask
            ],
                                  axis=1)

        return mask.reshape(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    # Modified transformers.models.llama.modeling_llama._expand_mask with np.array
    def _expand_mask(self, mask, dtype, tgt_len=None):
        """  
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.  
        """
        bsz, src_len = mask.shape
        if tgt_len is None:
            tgt_len = src_len
        # expand [bsz,38] to [bsz,1,1,38]
        expanded_mask = np.expand_dims(mask, axis=1)
        expanded_mask = np.expand_dims(mask, axis=1)
        expanded_mask = np.broadcast_to(expanded_mask,
                                        (bsz, 1, tgt_len, src_len))
        inverted_mask = 1.0 - expanded_mask

        cond = inverted_mask > 0
        return np.ma.array(inverted_mask,
                           mask=cond,
                           fill_value=np.finfo(dtype).min).filled()

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape,
                                        inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]

        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = self._make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = self._expand_mask(attention_mask,
                                                   inputs_embeds.dtype,
                                                   tgt_len=input_shape[-1])
            combined_attention_mask = (expanded_attn_mask
                                       if combined_attention_mask is None else
                                       expanded_attn_mask +
                                       combined_attention_mask)

        return combined_attention_mask

    def decode_init(self, input_ids: np.array):
        # project to embed/higher dimension
        hidden = self.init.embed(input_ids)
        assert hidden.shape[-1] == 4096

        seqlen = hidden.shape[1]
        position_ids = np.arange(seqlen, dtype=np.int64).reshape((1, seqlen))
        attention_mask = np.ones((1, seqlen), dtype=np.float32)
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (1, seqlen), hidden, 0)

        for idx in range(self.DECODER_COUNT):
            inputs = {
                'hidden_in': hidden,
                'attn_mask': attention_mask,
                'position_ids': position_ids
            }

            outputs = self.init.decode(inputs, idx)

            hidden = outputs['hidden_out']
            self.pastkeys[idx] = outputs['past_key']
            self.pastvalues[idx] = outputs['past_value']

        hidden = self.init.norm_head(hidden)
        return hidden

    def decode_past(self, token: np.array):
        # embed space
        hidden = self.past.embed(token)
        assert hidden.shape[-1] == 4096

        pastlen = self.pastkeys[0].shape[-2]
        seqlen = hidden.shape[1]
        assert seqlen == 1

        position_ids = np.arange(seqlen, dtype=np.int64).reshape((1, seqlen))
        position_ids[0][0] = pastlen

        attention_mask = np.ones((1, seqlen + pastlen), dtype=np.float32)
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (1, seqlen), hidden, pastlen)

        for idx in range(self.DECODER_COUNT):
            past_key = self.pastkeys[idx]
            past_value = self.pastvalues[idx]

            inputs = {
                'hidden_in': hidden,
                'attn_mask': attention_mask,
                'position_ids': position_ids,
                'past_key_in': past_key,
                'past_value_in': past_value
            }

            outputs = self.past.decode(inputs, idx)

            hidden = outputs[
                'hidden_out']  # [[[ 0.0221,  0.0120,  0.0007,  ..., -0.0614, -0.0625,  0.0494]]]
            self.pastkeys[idx] = outputs['past_key']
            self.pastvalues[idx] = outputs['past_value']

        hidden = self.init.norm_head(hidden)
        return hidden

    def apply_warp(self, tensor: np.array):
        tensor = warp_temperature(tensor, self.config['temperature'])
        tensor = warp_topk(tensor, self.config['topk'])
        return tensor

    def sample(self, prompt: str = 'bonjour'):
        prompt = prompt.strip()
        format_prompt = PROMPT.format_map({'instruction': prompt})

        # no EOS
        input_ids = self.tokenizer.encode(format_prompt, True, False)
        input_ids = np.array(input_ids, dtype=np.int64).reshape(
            (1, len(input_ids)))

        # decoder backbone loop
        while True:
            if self.pastkeys is None:
                # init cache
                self.pastkeys = [None for i in range(self.DECODER_COUNT)]
                self.pastvalues = [None for i in range(self.DECODER_COUNT)]
                # decoder backbone init
                logits = self.decode_init(input_ids)

            else:

                logits = self.decode_past(next_token)

            # split tail
            next_token_scores = logits[:, -1, :]

            # wrap logits for better output
            next_token_scores = self.apply_warp(next_token_scores)

            probs = npsoftmax(next_token_scores.astype(np.float64), axis=1)

            # Caution:
            # *** ValueError: sum(pvals[:-1].astype(np.float64)) > 1.0. The pvals array is cast to 64-bit floating point prior to checking the sum. Precision changes when casting may cause problems even if the sum of the original pvals is valid.
            next_token = npmultinominal2D(probs).astype(input_ids.dtype)
            logger.debug(next_token)

            input_ids = np.concatenate(
                [input_ids, next_token.reshape((1, 1))], axis=1)

            if input_ids.shape[-1] >= self.config['max'] or next_token[
                    0, 0] == self.FINISH_TOKEN:
                break

        # decode
        decoded = self.tokenizer.decode(input_ids[0].tolist())
        out = str(decoded.split('Response:')[1])
        logger.debug('Q: {} A: {}'.format(prompt, out))
        return out


def parse_args():
    parser = argparse.ArgumentParser(description='llama.onnx onnxruntime demo')
    parser.add_argument('onnxdir', help='llama 7B onnx model directory.')
    parser.add_argument('prompt', help='prompt text.')
    parser.add_argument(
        '--temperature',
        default=0.1,
        type=float,
        help=
        'factor to scale up logits, 1.0 means no warp. use `0.1` by default.')
    parser.add_argument(
        '--topk',
        default=40,
        type=int,
        help=
        'filter k high score values from logits, None means no filter. 40 by default.'
    )
    parser.add_argument(
        '--max',
        default=2000,
        type=int,
        help=
        'stop condition. default value is 2000, it would stop until len(output_token)==2000.'
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    llama = Llama(onnxdir=args.onnxdir,
                  config={
                      'temperature': args.temperature,
                      'topk': args.topk,
                      'max': args.max
                  })
    llama.sample(args.prompt)


if __name__ == '__main__':
    main()
