########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import numpy as np
import os
import shutil
import onnx
import onnxsim
import argparse
from loguru import logger
from onnxconverter_common import float16

np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch
from torch.nn import functional as F
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("20B_tokenizer.json")

args = types.SimpleNamespace()
args.MODEL_NAME = '/models/rwkv-4-pile-430m/RWKV-4-Pile-430M-20220808-8066'
args.n_layer = 24
args.n_embd = 1024

context = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
NUM_TRIALS = 3
LENGTH_PER_TRIAL = 100
TEMPERATURE = 1.0
TOP_P = 0.85

########################################################################################################

def onnx_fp32_to_fp16(onnxfile: str):
    fp32_model = onnx.load(onnxfile)

    inferred_model, check = onnxsim.simplify(fp32_model)
    assert check == True

    fp16_model = float16.convert_float_to_float16(inferred_model)
    onnx.save(fp16_model, onnxfile)


class Encoder(torch.nn.Module):
    def __init__(self, emb, ln_weight, ln_bias):
        super().__init__()
        self.emb = emb
        self.ln_weight = ln_weight
        self.ln_bias = ln_bias
    
    def forward(self, token):
        x = self.emb[token]
        x = F.layer_norm(x, (1024, ), weight=self.ln_weight, bias=self.ln_bias)
        x= x.flatten()
        return x


class Decoder(torch.nn.Module):
    def __init__(self, head, ln_weight, ln_bias):
        super().__init__()
        self.head = head
        self.ln_weight = ln_weight
        self.ln_bias = ln_bias
    
    def forward(self, x):
        x = F.layer_norm(x, (1024, ), weight=self.ln_weight, bias=self.ln_bias)
        x = self.head @ x
        return x.flatten()


class Mixer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def set_ffn(self, time_mix_k, time_mix_r, kw, vw, rw, ln_weight, ln_bias):
        self.ffn_time_mix_k = time_mix_k
        self.ffn_time_mix_r = time_mix_r
        self.ffn_kw = kw
        self.ffn_vw = vw
        self.ffn_rw = rw
        self.ffn_ln_weight = ln_weight
        self.ffn_ln_bias = ln_bias

    def forward_ffn(self, x, state):
        y = x
        x = F.layer_norm(x, (1024, ), weight=self.ffn_ln_weight, bias=self.ffn_ln_bias)
        xk = x * self.ffn_time_mix_k + state[0].flatten() * (1 - self.ffn_time_mix_k)
        xr = x * self.ffn_time_mix_r + state[0].flatten() * (1 - self.ffn_time_mix_r)
        r = torch.sigmoid(self.ffn_rw @ xr)
        k = torch.square(torch.relu(self.ffn_kw @ xk))  # square relu, primer paper
        return y + (r * (self.ffn_vw @ k)), x.reshape(1, -1)

    def set_attn(self, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow, ln_weight, ln_bias):
        self.time_mix_k = time_mix_k
        self.time_mix_v = time_mix_v
        self.time_mix_r = time_mix_r
        self.time_first = time_first
        self.time_decay = time_decay

        self.kw = kw
        self.vw = vw
        self.rw = rw
        self.ow = ow

        self.ln_weight = ln_weight
        self.ln_bias = ln_bias

    def forward_att(self, x, state):
        y = x

        x = F.layer_norm(x, (1024, ), weight=self.ln_weight, bias=self.ln_bias)

        xk = x * self.time_mix_k + state[1].flatten() * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + state[1].flatten() * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + state[1].flatten() * (1 - self.time_mix_r)
        
        r = torch.sigmoid(self.rw @ xr)
        k = self.kw @ xk
        v = self.vw @ xv

        aa = state[2].flatten()
        bb = state[3].flatten()
        pp = state[4].flatten()
        ww = self.time_first + k
        qq = torch.maximum(pp, ww)
        e1 = torch.exp(pp - qq)
        e2 = torch.exp(ww - qq)
        a = e1 * aa + e2 * v
        b = e1 * bb + e2
        wkv = a / b
        ww = pp + self.time_decay
        qq = torch.maximum(ww, k)
        e1 = torch.exp(ww - qq)
        e2 = torch.exp(k - qq)

        s1 = x.reshape(1, -1)
        s2 = (e1 * aa + e2 * v).reshape(1, -1)
        s3 = (e1 * bb + e2).reshape(1, -1)
        s4 = qq.reshape(1, -1)
        # state_out = torch.cat([s1, s2, s3, s4])
        # state[5 * i + 1] = x
        # state[5 * i + 2] = e1 * aa + e2 * v
        # state[5 * i + 3] = e1 * bb + e2
        # state[5 * i + 4] = qq
        return y + (self.ow @ (r * wkv)), s1, s2, s3, s4

    def forward(self, x, state):
        x, s1, s2, s3, s4 = self.forward_att(x, state)
        x, s0 = self.forward_ffn(x, state)
        return x, torch.cat([s0,s1,s2,s3,s4])


class ChannelMixer(torch.nn.Module):
    def __init__(self, time_mix_k, time_mix_r, kw, vw, rw, ln_weight, ln_bias):
        super().__init__()
        self.time_mix_k = time_mix_k
        self.time_mix_r = time_mix_r
        self.kw = kw
        self.vw = vw
        self.rw = rw
        self.ln_weight = ln_weight
        self.ln_bias = ln_bias

    def forward(self, x, state):
        y = x
        x = F.layer_norm(x, (1024, ), weight=self.ln_weight, bias=self.ln_bias)
        xk = x * self.time_mix_k + state[0].flatten() * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + state[0].flatten() * (1 - self.time_mix_r)
        r = torch.sigmoid(self.rw @ xr)
        k = torch.square(torch.relu(self.kw @ xk))  # square relu, primer paper
        return y + (r * (self.vw @ k)), x



class RWKV_RNN(torch.jit.ScriptModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.eval()  # set torch to inference mode

        w = torch.load(args.MODEL_NAME + '.pth', map_location='cpu')
        for k in w.keys():
            if '.time_' in k: w[k] = w[k].squeeze()
            if '.time_decay' in k: w[k] = -torch.exp(w[k].float())  # the real time decay is like e^{-e^x}
            else: w[k] = w[k].float()  # convert to f32 type

        self.w = types.SimpleNamespace()  # set self.w from w
        self.w.blocks = {}
        for k in w.keys():  # example: "blocks.0.att.time_first" => self.w.blocks[0].att.time_first
            parts = k.split('.')
            last = parts.pop()
            here = self.w
            for p in parts:
                if p.isdigit():
                    p = int(p)
                    if p not in here: here[p] = types.SimpleNamespace()
                    here = here[p]
                else:
                    if not hasattr(here, p): setattr(here, p, types.SimpleNamespace())
                    here = getattr(here, p)
            setattr(here, last, w[k])

        self.encoder = Encoder(self.w.emb.weight, self.w.blocks[0].ln0.weight, self.w.blocks[0].ln0.bias)
        self.encoder.eval()

        self.decoder = Decoder(self.w.head.weight, self.w.ln_out.weight, self.w.ln_out.bias)
        self.decoder.eval()

    def layer_norm(self, x, w):
        return F.layer_norm(x, (self.args.n_embd, ), weight=w.weight, bias=w.bias)

    @torch.jit.script_method
    def channel_mixing(self, x, state, i: int, time_mix_k, time_mix_r, kw, vw, rw):
        xk = x * time_mix_k + state[5 * i + 0] * (1 - time_mix_k)
        xr = x * time_mix_r + state[5 * i + 0] * (1 - time_mix_r)
        state[5 * i + 0] = x
        r = torch.sigmoid(rw @ xr)
        k = torch.square(torch.relu(kw @ xk))  # square relu, primer paper
        return r * (vw @ k)

    @torch.jit.script_method
    def time_mixing(self, x, state, i: int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
        xk = x * time_mix_k + state[5 * i + 1] * (1 - time_mix_k)
        xv = x * time_mix_v + state[5 * i + 1] * (1 - time_mix_v)
        xr = x * time_mix_r + state[5 * i + 1] * (1 - time_mix_r)
        state[5 * i + 1] = x
        r = torch.sigmoid(rw @ xr)
        k = kw @ xk
        v = vw @ xv

        aa = state[5 * i + 2]
        bb = state[5 * i + 3]
        pp = state[5 * i + 4]
        ww = time_first + k
        qq = torch.maximum(pp, ww)
        e1 = torch.exp(pp - qq)
        e2 = torch.exp(ww - qq)
        a = e1 * aa + e2 * v
        b = e1 * bb + e2
        wkv = a / b
        ww = pp + time_decay
        qq = torch.maximum(ww, k)
        e1 = torch.exp(ww - qq)
        e2 = torch.exp(k - qq)
        state[5 * i + 2] = e1 * aa + e2 * v
        state[5 * i + 3] = e1 * bb + e2
        state[5 * i + 4] = qq
        return ow @ (r * wkv)


    @torch.jit.script_method
    def encode_export(self, token, emb, ln_weight, ln_bias):
        x = emb[token]
        x = F.layer_norm(x, (1024, ), weight=ln_weight, bias=ln_bias)
        return x.flatten()


    @torch.jit.script_method
    def decode_export(self, x, head, ln_weight, ln_bias):
        x = F.layer_norm(x, (1024, ), weight=ln_weight, bias=ln_bias)
        x = head @ x
        return x.flatten()

    def forward(self, tokenid, state):
        with torch.no_grad():

            # x = self.w.emb.weight[token]
            # x = self.layer_norm(x, self.w.blocks[0].ln0)

            token = torch.full([1], tokenid, dtype=torch.int32)

            onnx_inputs = [token]
            onnx_filepath = 'models/embed.onnx'
            onnx_inp_names = ['token']
            onnx_out_names = ['output']
            torch.onnx.export(model=self.encoder, args=onnx_inputs, f=onnx_filepath, verbose=False, input_names=onnx_inp_names, output_names=onnx_out_names, opset_version=16)
            onnx_fp32_to_fp16(onnx_filepath)

            x = self.encoder.forward(token)

            for i in range(self.args.n_layer):

                att = self.w.blocks[i].att
                ln1 = self.w.blocks[i].ln1

                ffn = self.w.blocks[i].ffn
                ln2 = self.w.blocks[i].ln2

                mixer = Mixer()
                mixer.set_attn(att.time_mix_k, att.time_mix_v, att.time_mix_r, att.time_first, att.time_decay, att.key.weight, att.value.weight, att.receptance.weight, att.output.weight, ln1.weight, ln1.bias)
                mixer.set_ffn(ffn.time_mix_k, ffn.time_mix_r, ffn.key.weight, ffn.value.weight, ffn.receptance.weight, ln2.weight, ln2.bias)
               
                state_slice = state[5*i : 5*(i+1)]

                onnx_inputs = (x, state_slice)
                onnx_filepath = 'models/mixing_{}.onnx'.format(i)
                onnx_inp_names = ('input', 'state_in')
                onnx_out_names = ('output', 'state_out')
                torch.onnx.export(model=mixer, args=onnx_inputs, f=onnx_filepath, verbose=False, input_names=onnx_inp_names, output_names=onnx_out_names, opset_version=16)
                onnx_fp32_to_fp16(onnx_filepath)

                x, state_out = mixer.forward(x, state_slice)
                state[5*i:5*(i+1)] = state_out

            onnx_inputs = [x]
            onnx_filepath = 'models/head.onnx'
            onnx_inp_names = ['x']
            onnx_out_names = ['output']
            torch.onnx.export(model=self.decoder, args=onnx_inputs, f=onnx_filepath, verbose=False, input_names=onnx_inp_names, output_names=onnx_out_names, opset_version=16)
            onnx_fp32_to_fp16(onnx_filepath)

            x = self.decoder.forward(x)

            import pdb
            pdb.set_trace()
            return x.float(), state


##########################################################################################################


def sample_logits(out, temperature=1.0, top_p=0.8):
    probs = F.softmax(out, dim=-1).numpy()
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    probs[probs < cutoff] = 0
    if temperature != 1.0:
        probs = probs.pow(1.0 / temperature)
    probs = probs / np.sum(probs)
    out = np.random.choice(a=len(probs), p=probs)
    return out


########################################################################################################

print(f'\nUsing CPU. Loading {args.MODEL_NAME} ...')
model = RWKV_RNN(args)

print(f'\nPreprocessing context (slow version. see v2/rwkv/model.py for fast version)')

init_state = torch.zeros(args.n_layer * 5, args.n_embd)
for i in range(args.n_layer):
    init_state[5 * i + 4] = -1e30  # -infinity

for token in tokenizer.encode(context).ids:
    init_out, init_state = model.forward(token, init_state)

for TRIAL in range(NUM_TRIALS):
    print(f'\n\n--[ Trial {TRIAL} ]-----------------', context, end="")
    all_tokens = []
    out_last = 0
    out, state = init_out.clone(), init_state.clone()
    for i in range(LENGTH_PER_TRIAL):
        token = sample_logits(out, TEMPERATURE, TOP_P)
        all_tokens += [token]
        tmp = tokenizer.decode(all_tokens[out_last:])
        if '\ufffd' not in tmp:  # only print when we have a valid utf-8 string
            print(tmp, end="", flush=True)
            out_last = i + 1
        out, state = model.forward(token, state)
print('\n')
