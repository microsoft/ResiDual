# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from fairseq.modules.transformer_layer import TransformerEncoderLayerBase, TransformerDecoderLayerBase

class ResiDualEncoderLayer(TransformerEncoderLayerBase):
    def __init__(self, cfg, return_fc=False):
        super().__init__(cfg, return_fc)
        

    def forward(self, data: Tuple[Tensor, Tensor], encoder_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor] = None):
        x, res = data
        if self.training:
            self.ever_training = True
        
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(
                    attn_mask.to(torch.bool), -1e8 if x.dtype == torch.float32 else -1e4
            )
        xr = x
        x, _ = self.self_attn(
            query = x,
            key = x,
            value = x,
            key_padding_mask = encoder_padding_mask,
            need_weights = False,
            attn_mask = attn_mask
        )

        x = self.dropout_module(x)

        res = res + x * self.cfg.enc_res_input_norm_scale 

        x = self.self_attn_layer_norm(x + xr)
        

        xr = x
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)

        fc_result = x

        x = self.dropout_module(x)
        res = res + x * self.cfg.enc_res_input_norm_scale
        x = self.final_layer_norm(x + xr)
        

        if self.return_fc and not torch.jit.is_scripting():
            return (x, res), fc_result
        return (x, res)

class ResiDualDecoderLayer(TransformerDecoderLayerBase):
    def __init__(self, cfg, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False):
        super().__init__(cfg, no_encoder_attn, add_bias_kv, add_zero_attn)
        self.cfg = cfg
    
    def forward(
        self,
        data,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        x, res = data

        if need_head_weights:
            need_attn = True
        
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x
        
        xr = x
        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        if self.c_attn is not None:
            tgt_len, bsz = x.size(0), x.size(1)
            x = x.view(tgt_len, bsz, self.nh, self.head_dim)
            x = torch.einsum("tbhd,h->tbhd", x, self.c_attn)
            x = x.reshape(tgt_len, bsz, self.embed_dim)
        if self.attn_ln is not None:
            x = self.attn_ln(x)

        x = self.dropout_module(x)
        res = res + x * self.cfg.dec_res_input_norm_scale #self.res_input_norms[0](x)
        x = self.self_attn_layer_norm(x + xr)
        

        if self.encoder_attn is not None and encoder_out is not None:
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            xr = x
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            res = res + x * self.cfg.dec_res_input_norm_scale #self.res_input_norms[1](x)
            x = self.encoder_attn_layer_norm(x + xr)
            
        xr = x
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)

        x = self.dropout_module(x)
        res = res + x * self.cfg.dec_res_input_norm_scale # self.res_input_norms[2](x)
        x = self.final_layer_norm(x+xr)
        

        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return (x, res), attn, self_attn_state
        return (x, res), attn, None
