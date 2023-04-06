from llama import Tokenizer, Embed, Decoder
import numpy as np
import os
import pdb

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
PROMPT = PROMPT_DICT['prompt_no_input']

class Llama:
    def __init__(self, config: dict = {}):
        text = 'bonjour'
        # for AX
        onnxdir = '/nvme/konghuanjun/llama.onnx/'

        self.tokenizer = Tokenizer(os.path.join(onnxdir, 'tokenizer.model'))
        self.embed = Embed(onnxdir)
        self.DECODER_COUNT = 32
        self.init = Decoder(onnxdir, 'decoder-{}.onnx', self.DECODER_COUNT)
        self.past = None
        # self.past = Decoder(onnxdir, 'decoder-past-{}.onnx', self.DECODER_COUNT)
        self.config = config

        # cache
        self.pastkeys = [ None for i in range(self.DECODER_COUNT)]
        self.pastvalues = [ None for i in range(self.DECODER_COUNT)]


    # Modified transformers.models.llama.modeling_llama._make_causal_mask with np.array
    def _make_causal_mask(self, input_ids_shape, dtype, past_key_values_length: int=0):    
        """    
        Make causal mask used for bi-directional self-attention. 
        Output triangle-matrix if `past_key_values_length`=0
        Padding left if `past_key_values_length`>0
        """    
        bsz, tgt_len = input_ids_shape    
        mask = np.full((tgt_len, tgt_len), fill_value = np.finfo(dtype).min)
        
        mask_cond = np.arange(mask.shape[1])    
        cond = mask_cond < (mask_cond + 1).reshape(-1,1)
        mask = np.ma.array(mask, mask=cond, fill_value=0).filled()
        # masked_fill_result = np.ma.masked_fill_(mask, condition_row_array)

        if past_key_values_length > 0:    
            mask = np.concatenate([np.zeros((tgt_len, past_key_values_length), dtype=dtype), mask], axis=1)

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
        expanded_mask = np.broadcast_to(expanded_mask, (bsz, 1, tgt_len, src_len))
        inverted_mask = 1.0 - expanded_mask

        cond = inverted_mask > 0
        return np.ma.array(inverted_mask, mask=cond, fill_value=np.finfo(dtype).min).filled()


    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
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
            expanded_attn_mask = self._expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask


    def decoder_init(self, hidden: np.array):
        seqlen = hidden.shape[1]
        position_ids = np.arange(seqlen, dtype=np.int64).reshape((1, seqlen))
        attention_mask = np.ones((1, seqlen), dtype=np.float32)
        attention_mask = self._prepare_decoder_attention_mask(attention_mask, (1, seqlen), hidden, 0)

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

        pdb.set_trace()
        hidden = self.init.norm_head(hidden)
        return hidden


    def decoder_past(self, hidden: np.array):
        # TODO
        seqlen = hidden.shape[1]
        position_ids = np.arange(seqlen, dtype=np.int64).reshape((1, seqlen))

        attention_mask = np.ones((1, seqlen), dtype=np.float32)

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


    def generate(self, prompt: str = 'bonjour'):
        prompt = prompt.strip()
        format_prompt = PROMPT.format_map({'instruction': prompt})

        # no EOS
        input_ids = self.tokenizer.encode(format_prompt, True, False)
        input_ids = np.array(input_ids, dtype=np.int64).reshape((1, len(input_ids)))

        # project to embed/higher dimension
        input_embed = self.embed.forward({'input': input_ids})['embed']
        assert input_embed.shape[-1] == 4096

        # decoder backbone init
        hidden = self.decoder_init(input_embed)

        # decoder backbone loop
        while hidden.shape[1] < 2000:
            hidden = self.decoder_past(hidden[:-1:])

            # TODO kinds of waper not supported

        # head
        logits = self.head.forward({'input': hidden})['output']
        assert logits.shape[-1] ==  32000 # vocab_size

        # decode
        echo = self.tokenizer.decode(logits)
        return echo


llama = Llama()
echo = llama.generate('bonjour')
print(echo)
