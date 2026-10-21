import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from einops import rearrange
from PIL import Image
from scipy.ndimage import gaussian_filter

from datasets import *
from MaPLe.ad_prompts import *
from kan.KANLayer import *
from clip import clip
from utils.load_save_model import *
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):

        pos_distance = torch.sum((anchor - positive).pow(2), dim=1)

        neg_distance = torch.sum((anchor - negative).pow(2), dim=1)

        loss = torch.relu(pos_distance - neg_distance + self.margin)

        return torch.mean(loss)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final # layer normalize 最终层标准化
        self.text_projection = clip_model.text_projection # 映射与图像维度一致
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class KanAttention(nn.Module):
    def __init__(self, in_dim, out_dim, channels, num_heads=6):
        super(KanAttention, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.proj_in = nn.Sequential(*[KANLayer(1, 3) for _ in range(channels)])
        self.proj_out = nn.Linear(self.in_dim, self.out_dim)

    def apply_kan(self, x_in_cd):
        c, d = x_in_cd.shape
        x_in_dc = x_in_cd.permute(1, 0)

        q = []
        k = []
        v = []
        for i in range(c):
            xi = x_in_dc[:, i:i+1]
            xi = self.proj_in[i](xi)
            q.append(xi[:, 0:1])
            k.append(xi[:, 1:2])
            v.append(xi[:, 2:3])
        q = torch.cat(q, dim=1)
        k = torch.cat(k, dim=1)
        v = torch.cat(v, dim=1)
        q = q.permute(1, 0)
        k = k.permute(1, 0)
        v = v.permute(1, 0)
        
        return q, k, v
    
    def forward(self, x):
        q, k, v = self.apply_kan(x)
        q = rearrange(q, '(head c_head) d -> head c_head d', head=self.num_heads)
        k = rearrange(k, '(head c_head) d -> head c_head d', head=self.num_heads)
        v = rearrange(v, '(head c_head) d -> head c_head d', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'head c_head d -> (head c_head) d') + x
        out = self.proj_out(out)
        return out


class MLPPromptProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        hidden_dim = max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class TransformerPromptProjection(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(in_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(in_dim)
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, in_dim * 4),
            nn.GELU(),
            nn.Linear(in_dim * 4, in_dim),
        )
        self.norm2 = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x_seq = x.unsqueeze(0)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq, need_weights=False)
        x_seq = self.norm1(x_seq + attn_out)
        x_seq = self.norm2(x_seq + self.ffn(x_seq))
        return self.proj(x_seq.squeeze(0))


class MultiModalPromptLearner(nn.Module):
    def __init__(self, args, clip_model):
        super().__init__()
        n_ctx = args.n_ctx
        n_pro = args.n_pro
        n_pro_ab = args.n_pro_ab
        n_ctx_ab = args.n_ctx_ab

        use_generic_anomaly_prompt = getattr(args, 'use_generic_anomaly_prompt', True)
        use_category_anomaly_prompt = getattr(args, 'use_category_anomaly_prompt', True)
        use_learnable_anomaly_prompt = getattr(args, 'use_learnable_anomaly_prompt', True)

        state_anomaly1 = []
        self.n_generic_states = 0
        self.n_category_states = 0
        if use_generic_anomaly_prompt:
            state_anomaly1.extend(state_anomaly)
            self.n_generic_states = len(state_anomaly)
        if use_category_anomaly_prompt:
            category_states = class_state_abnormal.get(args.class_name, [])
            state_anomaly1.extend(category_states)
            self.n_category_states = len(category_states)
        if len(state_anomaly1) == 0 and not use_learnable_anomaly_prompt:
            raise ValueError("At least one anomaly prompt source must be enabled.")

        classname = args.class_name
        if args.class_name in class_mapping:
            classname = class_mapping[args.class_name]

        dtype = clip_model.dtype
        self.dtype = dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = args.input_size0
        
        assert args.prompt_depth >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = args.prompt_depth  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # random initialization
        normal_ctx_vectors = torch.empty(n_pro, n_ctx, ctx_dim, dtype=dtype, device=clip_model.token_embedding.weight.device)
        abnormal_ctx_vectors = torch.empty(n_pro_ab, n_ctx_ab, ctx_dim, dtype=dtype, device=clip_model.token_embedding.weight.device)

        nn.init.normal_(normal_ctx_vectors, std=0.02)
        nn.init.normal_(abnormal_ctx_vectors, std=0.02)

        normal_prompt_prefix = " ".join(["N"] * n_ctx)
        abnormal_prompt_prefix = " ".join(["A"] * n_ctx_ab)

        self.normal_ctx = nn.Parameter(normal_ctx_vectors)  # to be optimized
        self.abnormal_ctx = nn.Parameter(abnormal_ctx_vectors)  # to be optimized
        self.use_learnable_anomaly_prompt = use_learnable_anomaly_prompt

        # normal prompt
        normal_prompts = [normal_prompt_prefix + " " + classname + "." for _ in range(n_pro)]

        # abnormal prompt
        self.n_ab_handle = len(state_anomaly1)
        abnormal_prompts_handle = [normal_prompt_prefix + " " + state.format(classname) + "." for state in state_anomaly1 for _ in range(n_pro)]
        abnormal_prompts_learned = []
        if self.use_learnable_anomaly_prompt:
            abnormal_prompts_learned = [normal_prompt_prefix + " " + abnormal_prompt_prefix + " " + classname + "." for _ in range(n_pro_ab) for _ in range(n_pro)]

        tokenized_normal_prompts = clip.tokenize(normal_prompts).to(clip_model.token_embedding.weight.device)
        tokenized_abnormal_prompts_handle = self._tokenize_or_empty(abnormal_prompts_handle, clip_model)
        tokenized_abnormal_prompts_learned = self._tokenize_or_empty(abnormal_prompts_learned, clip_model)

        with torch.no_grad():
            normal_embedding = clip_model.token_embedding(tokenized_normal_prompts).type(dtype)
            abnormal_embedding_handle = self._embed_or_empty(tokenized_abnormal_prompts_handle, clip_model, ctx_dim)
            abnormal_embedding_learned = self._embed_or_empty(tokenized_abnormal_prompts_learned, clip_model, ctx_dim)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("normal_token_prefix", normal_embedding[:, :1, :])  # SOS
        self.register_buffer("normal_token_suffix", normal_embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.register_buffer("abnormal_token_prefix_handle", abnormal_embedding_handle[:, :1, :])  # SOS
        self.register_buffer("abnormal_token_suffix_handle", abnormal_embedding_handle[:, 1 + n_ctx:, :])  # CLS, EOS

        self.register_buffer("abnormal_token_prefix_learned", abnormal_embedding_learned[:, :1, :])  # SOS
        self.register_buffer("abnormal_token_suffix_learned", abnormal_embedding_learned[:, 1 + n_ctx + n_ctx_ab:, :])  # CLS, EOS

        self.n_pro = n_pro
        self.n_ctx = n_ctx
        self.n_pro_ab = n_pro_ab
        self.n_ctx_ab = n_ctx_ab
        self.tokenized_normal_prompts = tokenized_normal_prompts  # torch.Tensor
        self.tokenized_abnormal_prompts_handle = tokenized_abnormal_prompts_handle  # torch.Tensor
        self.tokenized_abnormal_prompts_learned = tokenized_abnormal_prompts_learned  # torch.Tensor
        self.n_generic_prompts = self.n_generic_states * n_pro
        self.n_category_prompts = self.n_category_states * n_pro
        self.n_learnable_prompts = len(abnormal_prompts_learned)

        # print('MaPLe design: Multi-modal Prompt Learning')
        # print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        self.prompt_module = getattr(args, 'prompt_module', 'kan')
        self.proj = self._build_visual_prompt_projector(
            self.prompt_module,
            ctx_dim,
            clip_model.visual.embed_dim,
            channels=n_pro * n_ctx,
            device=clip_model.token_embedding.weight.device,
        )
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, ctx_dim))
                                                      for _ in range(self.compound_prompts_depth - 1)]).to(clip_model.token_embedding.weight.device)
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = self._build_visual_prompt_projector(
            self.prompt_module,
            ctx_dim,
            clip_model.visual.embed_dim,
            channels=n_ctx,
            device=clip_model.token_embedding.weight.device,
        )
        if single_layer is None:
            self.compound_prompt_projections = nn.ModuleList()
        else:
            self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

    def _build_visual_prompt_projector(self, prompt_module, in_dim, out_dim, channels, device):
        if prompt_module == "none":
            return None
        if prompt_module == "mlp":
            module = MLPPromptProjection(in_dim, out_dim)
        elif prompt_module == "transformer":
            module = TransformerPromptProjection(in_dim, out_dim)
        elif prompt_module == "kan":
            module = KanAttention(in_dim=in_dim, out_dim=out_dim, channels=channels)
        else:
            raise ValueError(f"Unsupported prompt_module: {prompt_module}")
        return module.to(device).half()

    def _tokenize_or_empty(self, prompts, clip_model):
        device = clip_model.token_embedding.weight.device
        if len(prompts) == 0:
            return torch.empty(0, 77, dtype=torch.long, device=device)
        return torch.cat([clip.tokenize(p) for p in prompts]).to(device=device)

    def _embed_or_empty(self, tokenized_prompts, clip_model, ctx_dim):
        if tokenized_prompts.shape[0] == 0:
            return torch.empty(0, 77, ctx_dim, dtype=self.dtype, device=clip_model.token_embedding.weight.device)
        return clip_model.token_embedding(tokenized_prompts).type(self.dtype)

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        normal_ctx = self.normal_ctx#.to(self.dtype)
        abnormal_ctx = self.abnormal_ctx

        normal_prefix = self.normal_token_prefix
        normal_suffix = self.normal_token_suffix

        if normal_ctx.dim() == 2:
            normal_ctx = normal_ctx.unsqueeze(0).expand(self.n_pro, -1, -1)

        normal_prompts = self.construct_prompts(normal_ctx, normal_prefix, normal_suffix)

        # handle abnormal prompt
        n_ab_handle = self.n_ab_handle

        n_pro, n_ctx, dim = normal_ctx.shape
        if n_ab_handle > 0:
            normal_ctx1 = normal_ctx.unsqueeze(0).expand(n_ab_handle, -1, -1, -1).reshape(-1, n_ctx, dim)

            abnormal_prefix_handle = self.abnormal_token_prefix_handle
            abnormal_suffix_handle = self.abnormal_token_suffix_handle

            abnormal_prompts_handle = self.construct_prompts(normal_ctx1, abnormal_prefix_handle, abnormal_suffix_handle)
        else:
            abnormal_prompts_handle = normal_prompts.new_empty((0, normal_prompts.shape[1], normal_prompts.shape[2]))

        # learned abnormal prompt
        if self.use_learnable_anomaly_prompt and self.n_pro_ab > 0:
            abnormal_prefix_learned = self.abnormal_token_prefix_learned
            abnormal_suffix_learned = self.abnormal_token_suffix_learned
            abnormal_ctx = self.abnormal_ctx
            n_pro_ad, n_ctx_ad, dim_ad = abnormal_ctx.shape
            normal_ctx2 = normal_ctx.unsqueeze(0).expand(self.n_pro_ab, -1, -1, -1).reshape(-1, n_ctx, dim)
            abnormal_ctx = abnormal_ctx.unsqueeze(0).expand(self.n_pro, -1, -1, -1).reshape(-1, n_ctx_ad, dim_ad)

            abnormal_prompts_learned = torch.cat(
                [
                    abnormal_prefix_learned,        # (n_pro * n_pro_ab, 1, dim)
                    normal_ctx2,                    # (n_pro * n_pro_ab, n_ctx, dim)
                    abnormal_ctx,                   # (n_pro * n_pro_ab, n_ctx_ab, dim)
                    abnormal_suffix_learned,        # (n_pro * n_pro_ab, *, dim)
                ],
                dim=1,
            )
        else:
            abnormal_prompts_learned = normal_prompts.new_empty((0, normal_prompts.shape[1], normal_prompts.shape[2]))

        # Before returning, need to transform
        # prompts to 896 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            layer_dtype = next(layer.parameters()).dtype
            visual_deep_prompts.append(layer(self.compound_prompts_text[index].to(dtype=layer_dtype)))
        # Now the other way around
        # We will project the textual prompts from 512 to 896

        # ctx_to_project = torch.cat(
        #     [(normal_ctx.reshape(-1, normal_ctx.shape[-1])), abnormal_ctx.reshape(-1, abnormal_ctx.shape[-1])], 
        #     dim=0,
        # )   # normal和abnormal的都有
        ctx_to_project = normal_ctx.reshape(-1, normal_ctx.shape[-1])   # 只有normal的
        projection = None
        if self.proj is not None:
            projection = self.proj(ctx_to_project)   # 投影到896
        # projection = None

        return [normal_prompts, abnormal_prompts_handle, abnormal_prompts_learned], \
                projection, self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 896 is required


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class CustomCLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype


def _convert_to_rgb(image):
    return image.convert('RGB')

class MaPLe(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        if torch.cuda.is_available():
            self.device = kwargs['device']
        else:
            self.device = torch.device("cpu")

        self.shot = kwargs['k_shot']

        self.out_size_h = kwargs['resolution']
        self.out_size_w = kwargs['resolution']

        self.precision = 'fp16' #precision  -40% GPU memory (2.8G->1.6G) with slight performance drop

        # version v1: no norm for each of linguistic embedding
        # version v2:    norm for each of linguistic embedding
        self.version = 'V1'
        self.use_text_branch = getattr(args, 'use_text_branch', True)
        self.use_visual_memory = getattr(args, 'use_visual_memory', True)
        self.fusion_type = getattr(args, 'fusion_type', 'harmonic')
        self.fusion_alpha = getattr(args, 'fusion_alpha', 0.5)
        self.fusion_variant = getattr(args, 'fusion_variant', '')
        self.memory_layers = [x.strip() for x in getattr(args, 'memory_layers', 'layer3,layer8').split(',') if x.strip()]
        if not self.use_text_branch and not self.use_visual_memory:
            raise ValueError("At least one scoring branch must be enabled.")

        self.transform = transforms.Compose([
            transforms.Resize((kwargs['img_resize'], kwargs['img_resize']), Image.BICUBIC),
            transforms.CenterCrop(kwargs['img_cropsize']),
            _convert_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_train, std=std_train)])

        self.gt_transform = transforms.Compose([
            transforms.Resize((kwargs['img_resize'], kwargs['img_resize']), Image.NEAREST),
            transforms.CenterCrop(kwargs['img_cropsize']),
            transforms.ToTensor()])
        
        self.build_model(args)
                
    def build_model(self, args):
        
        # print(f"-------------------正在加载 CLIP (backbone: {args.backbone})-------------------")
        clip_model = load_clip(args)

        # print("-------------------开始创建自定义的 CLIP-------------------")
        self.custom_clip = CustomCLIP(clip_model)
        self.prompt_learner = MultiModalPromptLearner(args, clip_model)

        self.dtype = clip_model.dtype
        self.grid_size = self.custom_clip.image_encoder.grid_size

        visual_gallery1 = torch.zeros((self.shot*self.grid_size[0]*self.grid_size[1], self.custom_clip.image_encoder.embed_dim)).to(self.device)
        self.register_buffer("feature_gallery1", visual_gallery1)

        visual_gallery2 = torch.zeros((self.shot*self.grid_size[0]*self.grid_size[1], self.custom_clip.image_encoder.embed_dim)).to(self.device)
        self.register_buffer("feature_gallery2", visual_gallery2)

        visual_gallery3 = torch.zeros((self.shot*self.grid_size[0]*self.grid_size[1], self.custom_clip.image_encoder.output_dim)).to(self.device)
        self.register_buffer("feature_gallery3", visual_gallery3)

        text_features = torch.zeros((2, self.custom_clip.image_encoder.output_dim)).to(self.device)
        self.register_buffer("text_features", text_features)

        if self.precision == 'fp16':
            self.feature_gallery1  = self.feature_gallery1.half()
            self.feature_gallery2  = self.feature_gallery2.half()
            self.feature_gallery3  = self.feature_gallery3.half()
            self.text_features  = text_features.half()

        # # for testing
        self.tokenized_normal_prompts = self.prompt_learner.tokenized_normal_prompts
        self.tokenized_abnormal_prompts_handle = self.prompt_learner.tokenized_abnormal_prompts_handle
        self.tokenized_abnormal_prompts_learned = self.prompt_learner.tokenized_abnormal_prompts_learned
        self.tokenized_abnormal_prompts = torch.cat([
            self.tokenized_abnormal_prompts_handle,
            self.tokenized_abnormal_prompts_learned,
        ], dim=0)

        # print("-------------------关闭图像和文本编码器的梯度-------------------")
        for _, param in self.custom_clip.named_parameters():
            param.requires_grad_(False)

        self.custom_clip.to(self.device)

    @torch.no_grad()
    def encode_image(self, image, shared_ctx, deep_compound_prompts_vision, is_eval=False):
        if is_eval:
            shared_ctx = None
            deep_compound_prompts_vision = None
        image_features, tokens, _, _ = self.custom_clip.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        tokens = tokens[:, :self.custom_clip.image_encoder.grid_size[0]*self.custom_clip.image_encoder.grid_size[1], :] \
            / tokens[:, :self.custom_clip.image_encoder.grid_size[0]*self.custom_clip.image_encoder.grid_size[1], :].norm(dim=-1, keepdim=True)
        feature1 = self.custom_clip.image_encoder.mid_feature1[:, :self.custom_clip.image_encoder.grid_size[0]*self.custom_clip.image_encoder.grid_size[1], :] \
             / self.custom_clip.image_encoder.mid_feature1[:, :self.custom_clip.image_encoder.grid_size[0]*self.custom_clip.image_encoder.grid_size[1], :].norm(dim=-1, keepdim=True)
        feature2 = self.custom_clip.image_encoder.mid_feature2[:, :self.custom_clip.image_encoder.grid_size[0]*self.custom_clip.image_encoder.grid_size[1], :] \
            / self.custom_clip.image_encoder.mid_feature2[:, :self.custom_clip.image_encoder.grid_size[0]*self.custom_clip.image_encoder.grid_size[1], :].norm(dim=-1, keepdim=True)
        tokens = tokens / tokens.norm(dim=-1, keepdim=True)

        return image_features, \
            tokens, \
            feature1, \
            feature2

    def encode_text_embedding(self, text_embedding, original_tokens, deep_compound_prompts_text):
        text_features = self.custom_clip.text_encoder(text_embedding, original_tokens, deep_compound_prompts_text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def encode_abnormal_text_embedding(self, abnormal_text_embeddings_handle, abnormal_text_embeddings_learned, deep_compound_prompts_text):
        abnormal_text_features = []
        if abnormal_text_embeddings_handle.shape[0] > 0:
            abnormal_text_features.append(self.encode_text_embedding(
                abnormal_text_embeddings_handle,
                self.tokenized_abnormal_prompts_handle,
                deep_compound_prompts_text,
            ))
        if abnormal_text_embeddings_learned.shape[0] > 0:
            abnormal_text_features.append(self.encode_text_embedding(
                abnormal_text_embeddings_learned,
                self.tokenized_abnormal_prompts_learned,
                deep_compound_prompts_text,
            ))
        if len(abnormal_text_features) == 0:
            raise ValueError("No abnormal text prompts are enabled.")
        return torch.cat(abnormal_text_features, dim=0)

    @torch.no_grad()
    def collect_anchor_diagnostics(self):
        [normal_text_embeddings, abnormal_text_embeddings_handle, abnormal_text_embeddings_learned], _, deep_compound_prompts_text, _ = self.prompt_learner()
        normal_text_features = self.encode_text_embedding(
            normal_text_embeddings,
            self.tokenized_normal_prompts,
            deep_compound_prompts_text,
        )

        generic_features = normal_text_features.new_empty((0, normal_text_features.shape[-1]))
        category_features = normal_text_features.new_empty((0, normal_text_features.shape[-1]))
        if abnormal_text_embeddings_handle.shape[0] > 0:
            handle_features = self.encode_text_embedding(
                abnormal_text_embeddings_handle,
                self.tokenized_abnormal_prompts_handle,
                deep_compound_prompts_text,
            )
            n_generic = self.prompt_learner.n_generic_prompts
            n_category = self.prompt_learner.n_category_prompts
            generic_features = handle_features[:n_generic]
            category_features = handle_features[n_generic:n_generic + n_category]

        learnable_features = normal_text_features.new_empty((0, normal_text_features.shape[-1]))
        if abnormal_text_embeddings_learned.shape[0] > 0:
            learnable_features = self.encode_text_embedding(
                abnormal_text_embeddings_learned,
                self.tokenized_abnormal_prompts_learned,
                deep_compound_prompts_text,
            )

        abnormal_groups = [generic_features, category_features, learnable_features]
        abnormal_features = torch.cat([group for group in abnormal_groups if group.shape[0] > 0], dim=0)
        normal_anchor = F.normalize(normal_text_features.mean(dim=0, keepdim=True), dim=-1)
        abnormal_anchor = F.normalize(abnormal_features.mean(dim=0, keepdim=True), dim=-1)
        anchor_distance = torch.norm(abnormal_anchor - normal_anchor, p=2, dim=-1)

        return {
            'normal_prompt_features': normal_text_features.detach().float().cpu(),
            'generic_abnormal_prompt_features': generic_features.detach().float().cpu(),
            'category_abnormal_prompt_features': category_features.detach().float().cpu(),
            'learnable_abnormal_prompt_features': learnable_features.detach().float().cpu(),
            'normal_anchor': normal_anchor.detach().float().cpu(),
            'abnormal_anchor': abnormal_anchor.detach().float().cpu(),
            'anchor_distance': anchor_distance.detach().float().cpu(),
        }

    @torch.no_grad()
    def build_text_feature_gallery(self):
        [normal_text_embeddings, abnormal_text_embeddings_handle, abnormal_text_embeddings_learned], _, deep_compound_prompts_text, _ = self.prompt_learner()

        if self.version == "V1":
            normal_text_features = self.encode_text_embedding(normal_text_embeddings, self.tokenized_normal_prompts, deep_compound_prompts_text)
            abnormal_text_features = self.encode_abnormal_text_embedding(abnormal_text_embeddings_handle, abnormal_text_embeddings_learned, deep_compound_prompts_text)
        elif self.version == "V2":
            normal_text_features = []
            for phrase_id in range(normal_text_embeddings.size()[0]):
                normal_text_feature = self.encode_text_embedding(normal_text_embeddings[phrase_id].unsqueeze(0), self.tokenized_normal_prompts, deep_compound_prompts_text)
                normal_text_feature = normal_text_feature/normal_text_feature.norm(dim=-1, keepdim=True)
                normal_text_features.append(normal_text_feature)
            normal_text_features = torch.cat(normal_text_features, 0).half()
            abnormal_text_features = []
            for abnormal_text_embeddings, tokenized_abnormal_prompts in [
                (abnormal_text_embeddings_handle, self.tokenized_abnormal_prompts_handle),
                (abnormal_text_embeddings_learned, self.tokenized_abnormal_prompts_learned),
            ]:
                for phrase_id in range(abnormal_text_embeddings.size()[0]):
                    abnormal_text_feature = self.encode_text_embedding(abnormal_text_embeddings[phrase_id].unsqueeze(0), tokenized_abnormal_prompts[phrase_id].unsqueeze(0), deep_compound_prompts_text)
                    abnormal_text_feature = abnormal_text_feature/abnormal_text_feature.norm(dim=-1, keepdim=True)
                    abnormal_text_features.append(abnormal_text_feature)
            abnormal_text_features = torch.cat(abnormal_text_features, 0).half()
        else:
            raise NotImplementedError

        avr_normal_text_features = torch.mean(normal_text_features, dim=0, keepdim=True)
        avr_abnormal_text_features = torch.mean(abnormal_text_features, dim=0, keepdim=True)

        avr_normal_text_features = avr_normal_text_features
        avr_abnormal_text_features = avr_abnormal_text_features
        text_features = torch.cat([avr_normal_text_features, avr_abnormal_text_features], dim=0)
        self.text_features.copy_(text_features / text_features.norm(dim=-1, keepdim=True))

        return normal_text_features

    def build_image_feature_gallery(self, features1, features2, features3=None):
        _, _, d1 = features1.shape
        self.feature_gallery1.copy_(F.normalize(features1[:, :self.grid_size[0]*self.grid_size[1], :].reshape(-1, d1), dim=-1))   # 舍弃了后面两个的特征，后面两个是class token和visual prompt

        _, _, d2 = features2.shape
        self.feature_gallery2.copy_(F.normalize(features2[:, :self.grid_size[0]*self.grid_size[1], :].reshape(-1, d2), dim=-1))
        if features3 is not None:
            _, _, d3 = features3.shape
            self.feature_gallery3.copy_(F.normalize(features3[:, :self.grid_size[0]*self.grid_size[1], :].reshape(-1, d3), dim=-1))

    def calculate_textual_anomaly_score(self, visual_features, task):
        # t = 100
        t = self.custom_clip.logit_scale
        # t = self.t
        N = visual_features[1].shape[0]

        if task == 'seg':
            # ############################################## local tokens scores ############################
            # token_features = self.cross_attention(visual_features[1])
            token_features = visual_features[1]
            local_normality_and_abnormality_score = (t * token_features @ self.text_features.T).softmax(dim=-1)

            local_abnormality_score = torch.sigmoid(torch.log(local_normality_and_abnormality_score[:, :, 1]) - \
                                        torch.log(local_normality_and_abnormality_score[:, :, 0]))

            local_abnormality_score = torch.zeros((N, self.grid_size[0] * self.grid_size[1])) + local_abnormality_score.cpu()
            local_abnormality_score = local_abnormality_score.reshape((N, self.grid_size[0], self.grid_size[1])).unsqueeze(1)

            return local_abnormality_score.detach()

        elif task == 'cls':
            # ################################################ global cls token scores ##########################
            # global_feature = self.cross_attention(visual_features[0].unsqueeze(dim=1)).squeeze(dim=1)
            global_feature = visual_features[0]
            global_normality_and_abnormality_score = (t * global_feature @ self.text_features.T).softmax(dim=-1)

            global_abnormality_score = global_normality_and_abnormality_score[:, 1]
            global_abnormality_score = global_abnormality_score.cpu()

            return global_abnormality_score.detach().numpy()

        else:
            assert 'task error'

    def calculate_visual_anomaly_score(self, visual_features):
        N = visual_features[1].shape[0]

        scores = []
        if 'layer3' in self.memory_layers:
            score1, _ = (1.0 - visual_features[2] @ self.feature_gallery1.t()).min(dim=-1)
            scores.append(score1 / 2.0)
        if 'layer8' in self.memory_layers:
            score2, _ = (1.0 - visual_features[3] @ self.feature_gallery2.t()).min(dim=-1)
            scores.append(score2 / 2.0)
        if 'final' in self.memory_layers:
            score3, _ = (1.0 - visual_features[1] @ self.feature_gallery3.t()).min(dim=-1)
            scores.append(score3 / 2.0)
        if len(scores) == 0:
            raise ValueError(f"No valid memory layers selected: {self.memory_layers}")

        score = torch.zeros((N, self.grid_size[0] * self.grid_size[1])) + torch.stack(scores, dim=0).mean(dim=0).cpu()

        return score.reshape((N, self.grid_size[0], self.grid_size[1])).unsqueeze(1)

    def fuse_anomaly_scores(self, textual_score, visual_score):
        if self.use_text_branch and not self.use_visual_memory:
            return textual_score
        if self.use_visual_memory and not self.use_text_branch:
            return visual_score

        eps = 1e-8
        if self.fusion_type in ('none', 'mean'):
            return 0.5 * (textual_score + visual_score)
        if self.fusion_type == 'max':
            return torch.maximum(textual_score, visual_score)
        if self.fusion_type == 'harmonic':
            return 2.0 / (1.0 / (textual_score + eps) + 1.0 / (visual_score + eps))
        if self.fusion_type == 'weighted':
            alpha = float(self.fusion_alpha)
            return alpha * textual_score + (1.0 - alpha) * visual_score
        raise ValueError(f"Unsupported fusion_type: {self.fusion_type}")

    def visual_map_to_image_score(self, visual_anomaly_map):
        return visual_anomaly_map.flatten(1).max(dim=1).values

    def forward(self, images, task, is_eval=False):
        _, shared_ctx, _, deep_compound_prompts_vision = self.prompt_learner()
        visual_features = self.encode_image(images, shared_ctx, deep_compound_prompts_vision, is_eval)
        if task == 'seg':
            textual_anomaly_map = None
            visual_anomaly_map = None
            if self.use_text_branch:
                textual_anomaly_map = self.calculate_textual_anomaly_score(visual_features, 'seg')
            if self.use_visual_memory:
                visual_anomaly_map = self.calculate_visual_anomaly_score(visual_features)

            if textual_anomaly_map is None:
                anomaly_map1 = visual_anomaly_map
                anomaly_map2 = visual_anomaly_map
            elif visual_anomaly_map is None:
                anomaly_map1 = textual_anomaly_map
                anomaly_map2 = textual_anomaly_map
            elif not self.fusion_variant:
                anomaly_map1 = 1. / (1. / textual_anomaly_map + 1. / visual_anomaly_map)
                anomaly_map2 = 0.5 * (textual_anomaly_map + visual_anomaly_map)
            else:
                anomaly_map1 = self.fuse_anomaly_scores(textual_anomaly_map, visual_anomaly_map)
                anomaly_map2 = 0.5 * (textual_anomaly_map + visual_anomaly_map)

            anomaly_map1 = F.interpolate(anomaly_map1, size=(self.out_size_h, self.out_size_w), mode='bilinear', align_corners=False)
            am_pix = anomaly_map1.squeeze(1).detach().numpy()
            am_pix_list1 = []
            for i in range(am_pix.shape[0]):
                am_pix[i] = gaussian_filter(am_pix[i], sigma=4)
                am_pix_list1.append(am_pix[i])

            anomaly_map2 = F.interpolate(anomaly_map2, size=(self.out_size_h, self.out_size_w), mode='bilinear', align_corners=False)
            am_pix = anomaly_map2.squeeze(1).detach().numpy()
            am_pix_list2 = []
            for i in range(am_pix.shape[0]):
                am_pix[i] = gaussian_filter(am_pix[i], sigma=4)
                am_pix_list2.append(am_pix[i])

            return am_pix_list1, am_pix_list2

        elif task == 'cls':
            textual_anomaly = None
            if self.use_text_branch:
                textual_anomaly = torch.as_tensor(self.calculate_textual_anomaly_score(visual_features, 'cls'))

            if self.use_visual_memory:
                visual_anomaly_map = self.calculate_visual_anomaly_score(visual_features)
            else:
                visual_anomaly_map = textual_anomaly.reshape(-1, 1, 1, 1)

            anomaly_map = F.interpolate(visual_anomaly_map, size=(self.out_size_h, self.out_size_w), mode='bilinear',
                                        align_corners=False)

            am_pix = anomaly_map.squeeze(1).detach().numpy()

            am_pix_list = []

            for i in range(am_pix.shape[0]):
                am_pix_list.append(am_pix[i])

            visual_anomaly = self.visual_map_to_image_score(visual_anomaly_map).cpu()
            if not self.fusion_variant and textual_anomaly is not None:
                fused_anomaly = textual_anomaly.cpu()
            elif textual_anomaly is None:
                fused_anomaly = visual_anomaly
            elif not self.use_visual_memory:
                fused_anomaly = textual_anomaly.cpu()
            else:
                fused_anomaly = self.fuse_anomaly_scores(textual_anomaly.cpu(), visual_anomaly)

            am_img_list = []
            for i in range(fused_anomaly.shape[0]):
                am_img_list.append(float(fused_anomaly[i]))

            return am_img_list, am_pix_list
        else:
            assert 'task error'

    def train_mode(self):
        self.custom_clip.train()

    def eval_mode(self):
        self.custom_clip.eval()
