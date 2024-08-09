from typing import Dict, Tuple
import diffusers
from ldm.modules.diffusionmodules.util import timestep_embedding
from ddim.models.diffusion import AttnBlock, ResnetBlock, get_timestep_embedding, nonlinearity
from ldm.modules.diffusionmodules.openaimodel import AttentionBlock, QKMatMul, ResBlock, SMVMatMul, TimestepBlock, checkpoint
from ldm.modules.attention import BasicTransformerBlock
import torch as th
import torch.nn as nn
from torch import einsum
from ldm.modules.attention import exists, default, CrossAttention
from einops import rearrange, repeat
from types import MethodType
from quant.quant_layer import QuantLayer, UniformAffineQuantizer, StraightThrough
from diffusers.models.embeddings import TimestepEmbedding, PixArtAlphaCombinedTimestepSizeEmbeddings
from diffusers.models.normalization import AdaLayerNormSingle
from typing import Optional


class BaseQuantBlock(nn.Module):

    def __init__(self,
                 aq_params: dict = {}
                 ) -> None:
        super().__init__()
        self.use_wq = False
        self.use_aq = False
        self.act_func = StraightThrough()
        self.ignore_recon = False

    def set_quant_state(self,
                        use_wq: bool = False,
                        use_aq: bool = False
                        ) -> None:
        for m in self.modules():
            if isinstance(m, QuantLayer):
                m.set_quant_state(use_wq=use_wq, use_aq=use_aq)


class QuantTemporalInformationBlockPixArt(BaseQuantBlock):

    def __init__(self,
                 adaln: AdaLayerNormSingle,
                 aq_params: dict = {},
                 ) -> None:
        super().__init__(aq_params)
        self.emb = adaln.emb

        self.silu = adaln.silu
        self.linear = adaln.linear

    def forward(
        self,
        timestep: th.Tensor,
        added_cond_kwargs: Optional[Dict[str, th.Tensor]] = None,
        batch_size: Optional[int] = None,
        hidden_dtype: Optional[th.dtype] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        # No modulation happening here.
        embedded_timestep = self.emb(timestep, **added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep

    def set_quant_state(self,
                        use_wq: bool = False,
                        use_aq: bool = False
                        ) -> None:
        for m in self.modules():
            if isinstance(m, QuantLayer):
                m.set_quant_state(use_wq=use_wq, use_aq=use_aq)
        for emb_layer in self.emb_layers:
            for m in emb_layer.modules():
                if isinstance(m, QuantLayer):
                    m.set_quant_state(use_wq=use_wq, use_aq=use_aq)

class QuantTemporalInformationBlockDDIM(BaseQuantBlock):

    def __init__(self,
                 temb: nn.Module,
                 aq_params: dict = {},
                 ch: int = None
                 ) -> None:
        super().__init__(aq_params)
        self.temb = temb
        self.temb_projs = []
        self.ch = ch

    def add_temb_proj(self,
                      temb_proj: nn.Linear) -> None:
        self.temb_projs.append(temb_proj)

    def forward(self,
                x: th.Tensor,
                t: th.Tensor,
                ) -> Tuple[th.Tensor]:
        assert t is not None
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        opts = []
        for temb_proj in self.temb_projs:
            opts.append(temb_proj(nonlinearity(temb)))
        return tuple(opts)

    def set_quant_state(self,
                        use_wq: bool = False,
                        use_aq: bool = False
                        ) -> None:
        for m in self.modules():
            if isinstance(m, QuantLayer):
                m.set_quant_state(use_wq=use_wq, use_aq=use_aq)
        for temb_proj in self.temb_projs:
            assert isinstance(temb_proj, QuantLayer)
            temb_proj.set_quant_state(use_wq=use_wq, use_aq=use_aq)


class QuantTemporalInformationBlock(BaseQuantBlock):

    def __init__(self,
                 t_emb: nn.Sequential,
                 aq_params: dict = {},
                 model_channels: int = None,
                 num_classes: int = None
                 ) -> None:
        super().__init__(aq_params)
        self.t_emb = t_emb
        self.emb_layers = []
        self.label_emb_layer = None
        self.model_channels = model_channels
        self.num_classes = num_classes

    def add_emb_layer(self,
                      layer: nn.Sequential) -> None:
        self.emb_layers.append(layer)

    def add_label_emb_layer(self,
                            layer: nn.Sequential) -> None:
        self.label_emb = layer

    def forward(self,
                x: th.Tensor,
                t: th.Tensor,
                y: th.Tensor = None
                ) -> Tuple[th.Tensor]:
        assert t is not None
        t_emb = timestep_embedding(t, self.model_channels, repeat_only=False)
        emb = self.t_emb(t_emb)
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        opts = []
        for layer in self.emb_layers:
            opts.append(layer(emb))
        return tuple(opts)

    def set_quant_state(self,
                        use_wq: bool = False,
                        use_aq: bool = False
                        ) -> None:
        for m in self.modules():
            if isinstance(m, QuantLayer):
                m.set_quant_state(use_wq=use_wq, use_aq=use_aq)
        for emb_layer in self.emb_layers:
            for m in emb_layer.modules():
                if isinstance(m, QuantLayer):
                    m.set_quant_state(use_wq=use_wq, use_aq=use_aq)


# --------- Stable Diffusion Model -------- #
class QuantResBlock(BaseQuantBlock, TimestepBlock):
    def __init__(
        self,
        res: ResBlock,
        aq_params: dict = {}
        ) -> None:
        super().__init__(aq_params)
        self.channels = res.channels
        self.emb_channels = res.emb_channels
        self.dropout = res.dropout
        self.out_channels = res.out_channels
        self.use_conv = res.use_conv
        self.use_checkpoint = res.use_checkpoint
        self.use_scale_shift_norm = res.use_scale_shift_norm

        self.in_layers = res.in_layers

        self.updown = res.updown

        self.h_upd = res.h_upd
        self.x_upd = res.x_upd

        self.emb_layers = res.emb_layers
        self.out_layers = res.out_layers

        self.skip_connection = res.skip_connection

    def forward(self,
                x: th.Tensor,
                emb: th.Tensor = None,
                split: int = 0
                ) -> th.Tensor:
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if split != 0 and hasattr(self.skip_connection, 'split') and self.skip_connection.split == 0:
            # resblock_updown use Identity() as skip_connection
            return checkpoint(
                self._forward, (x, emb, split), self.parameters(), self.use_checkpoint
            )
        return checkpoint(
                self._forward, (x, emb), self.parameters(), self.use_checkpoint
            )

    def _forward(self,
                 x: th.Tensor,
                 emb: th.Tensor,
                 split: int = 0
                 ) -> th.Tensor:
        if emb is None:
            assert(len(x) == 2)
            x, emb = x
        assert x.shape[2] == x.shape[3]

        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        if split != 0:
            return self.skip_connection(x, split=split) + h
        return self.skip_connection(x) + h


def cross_attn_forward(self: CrossAttention,
                        x: th.Tensor,
                        context: th.Tensor = None,
                        mask: th.Tensor = None
                        ) -> th.Tensor:
    h = self.heads

    q = self.to_q(x)
    context = default(context, x)
    k = self.to_k(context)
    v = self.to_v(context)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    if self.use_aq:
        quant_q = self.aqtizer_q(q)
        quant_k = self.aqtizer_k(k)
        sim = einsum('b i d, b j d -> b i j', quant_q, quant_k) * self.scale
    else:
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

    if exists(mask):
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -th.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)
    attn = sim.softmax(dim=-1)

    if self.use_aq:
        out = einsum('b i j, b j d -> b i d', self.aqtizer_w(attn), self.aqtizer_v(v))
    else:
        out = einsum('b i j, b j d -> b i d', attn, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    return self.to_out(out)


class QuantBasicTransformerBlock(BaseQuantBlock):
    def __init__(
        self,
        tran: BasicTransformerBlock,
        aq_params: dict = {},
        softmax_a_bit: int = 8
        ) -> None:
        super().__init__(aq_params)
        self.attn1 = tran.attn1
        self.ff = tran.ff
        self.attn2 = tran.attn2

        self.norm1 = tran.norm1
        self.norm2 = tran.norm2
        self.norm3 = tran.norm3
        self.checkpoint =  False

        self.attn1.aqtizer_q = UniformAffineQuantizer(**aq_params)
        self.attn1.aqtizer_k = UniformAffineQuantizer(**aq_params)
        self.attn1.aqtizer_v = UniformAffineQuantizer(**aq_params)

        self.attn2.aqtizer_q = UniformAffineQuantizer(**aq_params)
        self.attn2.aqtizer_k = UniformAffineQuantizer(**aq_params)
        self.attn2.aqtizer_v = UniformAffineQuantizer(**aq_params)

        aq_params_w = aq_params.copy()
        aq_params_w['bits'] = softmax_a_bit
        aq_params_w['symmetric'] = False
        aq_params_w['always_zero'] = True
        self.attn1.aqtizer_w = UniformAffineQuantizer(**aq_params_w)
        self.attn2.aqtizer_w = UniformAffineQuantizer(**aq_params_w)
        self.attn1.forward = MethodType(cross_attn_forward, self.attn1)
        self.attn2.forward = MethodType(cross_attn_forward, self.attn2)
        self.attn1.use_aq = False
        self.attn2.use_aq = False

    def forward(self,
                x: th.Tensor,
                context: th.Tensor = None
                ) -> th.Tensor:
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self,
                 x: th.Tensor,
                 context: th.Tensor = None
                 ) -> th.Tensor:
        assert context is not None

        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context) + x
        x = self.ff(self.norm3(x)) + x
        return x


# --------- Latent Diffusion Model -------- #
class QuantQKMatMul(BaseQuantBlock):
    def __init__(
        self,
        aq_params: dict = {}
        ) -> None:
        super().__init__(aq_params)
        self.scale = None
        self.use_aq = False
        self.aqtizer_q = UniformAffineQuantizer(**aq_params)
        self.aqtizer_k = UniformAffineQuantizer(**aq_params)

    def forward(self,
                q: th.Tensor,
                k: th.Tensor
                ) -> th.Tensor:
        if self.use_aq:
            quant_q = self.aqtizer_q(q * self.scale)
            quant_k = self.aqtizer_k(k * self.scale)
            weight = th.einsum(
                "bct,bcs->bts", quant_q, quant_k
            )
        else:
            weight = th.einsum(
                "bct,bcs->bts", q * self.scale, k * self.scale
            )
        return weight


class QuantSMVMatMul(BaseQuantBlock):
    def __init__(
        self,
        aq_params: dict = {},
        softmax_a_bit: int = 8
        ) -> None:
        super().__init__(aq_params)
        self.use_aq = False
        self.aqtizer_v = UniformAffineQuantizer(**aq_params)
        aq_params_w = aq_params.copy()
        aq_params_w['bits'] = softmax_a_bit
        aq_params_w['symmetric'] = False
        aq_params_w['always_zero'] = True
        self.aqtizer_w = UniformAffineQuantizer(**aq_params_w)

    def forward(self,
                weight: th.Tensor,
                v: th.Tensor
                ) -> th.Tensor:
        if self.use_aq:
            a = th.einsum("bts,bcs->bct", self.aqtizer_w(weight), self.aqtizer_v(v))
        else:
            a = th.einsum("bts,bcs->bct", weight, v)
        return a


class QuantAttentionBlock(BaseQuantBlock):
    def __init__(
        self,
        attn: AttentionBlock,
        aq_params: dict = {}
        ) -> None:
        super().__init__(aq_params)
        self.channels = attn.channels
        self.num_heads = attn.num_heads
        self.use_checkpoint = attn.use_checkpoint
        self.norm = attn.norm
        self.qkv = attn.qkv

        self.attention = attn.attention

        self.proj_out = attn.proj_out

    def forward(self,
                x: th.Tensor
                ) -> th.Tensor:
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self,
                 x: th.Tensor
                 ) -> th.Tensor:
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


# --------- DDIM Model -------- #
class QuantResnetBlock(BaseQuantBlock):
    def __init__(
        self,
        res: ResnetBlock,
        aq_params: dict = {}
        ) -> None:
        super().__init__(aq_params)
        self.in_channels = res.in_channels
        self.out_channels = res.out_channels
        self.use_conv_shortcut = res.use_conv_shortcut

        self.norm1 = res.norm1
        self.conv1 = res.conv1
        self.temb_proj = res.temb_proj
        self.norm2 = res.norm2
        self.dropout = res.dropout
        self.conv2 = res.conv2
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = res.conv_shortcut
            else:
                self.nin_shortcut = res.nin_shortcut


    def forward(self,
                x: th.Tensor,
                temb: th.Tensor = None,
                split: int = 0
                ) -> None:
        if temb is None:
            assert(len(x) == 2)
            x, temb = x

        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            elif hasattr(self.nin_shortcut, 'split'):
                x = self.nin_shortcut(x, split=split)
            else:
                x = self.nin_shortcut(x)
        out = x + h
        return out


class QuantAttnBlock(BaseQuantBlock):
    def __init__(
        self,
        attn: AttnBlock,
        aq_params: dict = {},
        softmax_a_bit: int = 8
        ) -> None:
        super().__init__(aq_params)
        self.in_channels = attn.in_channels

        self.norm = attn.norm
        self.q = attn.q
        self.k = attn.k
        self.v = attn.v
        self.proj_out = attn.proj_out

        self.aqtizer_q = UniformAffineQuantizer(**aq_params)
        self.aqtizer_k = UniformAffineQuantizer(**aq_params)
        self.aqtizer_v = UniformAffineQuantizer(**aq_params)

        aq_params_w = aq_params.copy()
        aq_params_w['bits'] = softmax_a_bit
        aq_params_w['symmetric'] = False
        aq_params_w['always_zero'] = True
        self.aqtizer_w = UniformAffineQuantizer(**aq_params_w)


    def forward(self,
                x: th.Tensor
                ) -> th.Tensor:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h*w)
        if self.use_aq:
            q = self.aqtizer_q(q)
            k = self.aqtizer_k(k)
        w_ = th.bmm(q, k)
        w_ = w_ * (int(c)**(-0.5))
        w_ = nn.functional.softmax(w_, dim=2)

        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)
        if self.use_aq:
            v = self.aqtizer_v(v)
            w_ = self.aqtizer_w(w_)
        h_ = th.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        out = x + h_
        return out


def b2qb(use_aq: bool = False) -> Dict[nn.Module, BaseQuantBlock]:
    D = {
        ResBlock.__name__: QuantResBlock,
        BasicTransformerBlock.__name__: QuantBasicTransformerBlock,
        ResnetBlock.__name__: QuantResnetBlock,
        AttnBlock.__name__: QuantAttnBlock,
        diffusers.models.attention.BasicTransformerBlock.__name__: QuantDiffBTB
    }
    if use_aq:
        D[QKMatMul.__name__] = QuantQKMatMul
        D[SMVMatMul.__name__] = QuantSMVMatMul
    else:
        D[AttentionBlock.__name__] = QuantAttentionBlock
    return D

import diffusers
from typing import Optional, Any, Dict
from quant.quant_aware_attn_processors import QuantAttnProcessor

class QuantDiffBTB(BaseQuantBlock):
    def __init__(self, tran,#: diffusers.models.attention.BasicTransformerBlock,
                 act_quant_params: dict = {}, sm_abit: int = 8):
        super().__init__(act_quant_params)
        self.only_cross_attention = tran.only_cross_attention
        self.use_ada_layer_norm_zero = tran.use_ada_layer_norm_zero
        self.use_ada_layer_norm = tran.use_ada_layer_norm

        self.use_ada_layer_norm_single = tran.use_ada_layer_norm_single # new
        self.use_layer_norm = tran.use_layer_norm # new
        self.use_ada_layer_norm_continuous = tran.use_ada_layer_norm_continuous # new
        self.norm_type = tran.norm_type # new
        self.num_embeds_ada_norm = tran.num_embeds_ada_norm # new
        self.pos_embed = tran.pos_embed # new

        self.norm1 = tran.norm1
        self.attn1 = tran.attn1
        self.norm2 = tran.norm2
        self.attn2 = tran.attn2
        if hasattr(tran, "norm3"):
            self.norm3 = tran.norm3
        else:
            self.norm3 = self.norm2
        self.ff = tran.ff

        if hasattr(tran, "fuser"):
            self.fuser = tran.fuser
        if hasattr(tran, "scale_shift_table"):
            self.scale_shift_table = tran.scale_shift_table

        self._chunk_size = tran._chunk_size
        self._chunk_dim = tran._chunk_dim
        self.set_chunk_feed_forward = tran.set_chunk_feed_forward  # This is a function handle, not variable

        # Check that the Attention Processor is correct
        assert isinstance(self.attn1.processor, (diffusers.models.attention_processor.AttnProcessor, diffusers.models.attention_processor.AttnProcessor2_0)), "Need to implement a different attention processor"
        self.attn1.set_processor(QuantAttnProcessor())

        if self.attn2 is not None:
            assert isinstance(self.attn2.processor, (diffusers.models.attention_processor.AttnProcessor, diffusers.models.attention_processor.AttnProcessor2_0)), "Need to implement a different attention processor"
            self.attn2.set_processor(QuantAttnProcessor())

        self.checkpoint = False
        self.attn1.act_quantizer_q = UniformAffineQuantizer(**act_quant_params)
        self.attn1.act_quantizer_k = UniformAffineQuantizer(**act_quant_params)
        self.attn1.act_quantizer_v = UniformAffineQuantizer(**act_quant_params)

        if self.attn2 is not None:
            self.attn2.act_quantizer_q = UniformAffineQuantizer(**act_quant_params)
            self.attn2.act_quantizer_k = UniformAffineQuantizer(**act_quant_params)
            self.attn2.act_quantizer_v = UniformAffineQuantizer(**act_quant_params)

        act_quant_params_w = act_quant_params.copy()
        act_quant_params_w['bits'] = sm_abit
        act_quant_params_w['always_zero'] = True
        self.attn1.act_quantizer_w = UniformAffineQuantizer(**act_quant_params_w)

        self.attn1.use_aq = False
        
        if self.attn2 is not None:
            self.attn2.act_quantizer_w = UniformAffineQuantizer(**act_quant_params_w)
            self.attn2.use_aq = False

    def set_quant_state(self, use_wq: bool = False, use_aq: bool = False):
        self.attn1.use_aq = use_aq
        if self.attn2 is not None:
            self.attn2.use_aq = use_aq

        self.use_wq = use_wq
        self.use_aq = use_aq
        for m in self.modules():
            if isinstance(m, QuantLayer):
                m.set_quant_state(use_wq, use_aq)

    def forward(self, 
                hidden_states,
                attention_mask: Optional[th.Tensor] = None,
                encoder_hidden_states: Optional[th.Tensor] = None,
                encoder_attention_mask: Optional[th.Tensor] = None,
                timestep: Optional[th.LongTensor] = None,
                cross_attention_kwargs: Dict[str, Any] = None,
                class_labels: Optional[th.LongTensor] = None,
                added_cond_kwargs: Optional[Dict[str, th.Tensor]] = None):
    
        if len(self._forward_hooks) > 0:
            self.am_cache = attention_mask
            self.ehs_cache = encoder_hidden_states
            self.eam_cache = encoder_attention_mask
            self.ts_cache = timestep
            self.cak_cache = cross_attention_kwargs
            self.class_labels = class_labels
            self.added_cond_kwargs = added_cond_kwargs
        else:
            self.am_cache = None
            self.ehs_cache = None
            self.eam_cache = None
            self.ts_cache = None
            self.cak_cache = None
            self.class_labels = None
            self.added_cond_kwargs = added_cond_kwargs
        return checkpoint(self._forward, (hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, timestep, cross_attention_kwargs, class_labels, added_cond_kwargs), self.parameters(), self.checkpoint)
    
    # A direct copy of Diffusers v0.19.0 BasicTransformerBlock forward function
    # https://github.com/huggingface/diffusers/blob/v0.19.0/src/diffusers/models/attention.py#L28
    # With minor changes to facilitate activation quantization of intermediate tensors, e.g, QK and SMV.
    def _forward(
        self,
        hidden_states,   # Not None
        attention_mask: Optional[th.FloatTensor] = None,  # None
        encoder_hidden_states: Optional[th.FloatTensor] = None, # Not None
        encoder_attention_mask: Optional[th.FloatTensor] = None, # None
        timestep: Optional[th.LongTensor] = None, # Not None
        cross_attention_kwargs: Dict[str, Any] = None, # None
        class_labels: Optional[th.LongTensor] = None, # None
        added_cond_kwargs: Optional[Dict[str, th.Tensor]] = None  # None
    ):
        
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")
        
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.norm_type == "ada_norm_zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif self.norm_type == "ada_norm_single":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            norm_hidden_states = norm_hidden_states.squeeze(1)
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

        if self.norm_type == "ada_norm_zero":
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.norm_type == "ada_norm_single":
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 1.2 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.norm_type == "ada_norm_single":
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states
        
        # 4. Feed-forward
        # i2vgen doesn't have this norm 🤷‍♂️
        if self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif not self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm3(hidden_states)

        if self.norm_type == "ada_norm_zero":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.norm_type == "ada_norm_zero":
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states