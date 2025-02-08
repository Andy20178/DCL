from typing import Tuple, Union
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math
import sys
from model.VAE.utils import contrastive_loss, compute_CDSVAE, compute_TransVAE
from model.VAE.TrasVAE import TransVAE
from model.VAE.CDSVAE import CDSVAE
import wandb
from model.utils import ModifiedResNet, VisionTransformer, Transformer, LayerNorm, knowledge_caluate
import time 

class PACSImageAudioCLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 # combined
                 dropout : float = 0.3,
                 audio_model = None,
                 args = None
                 ):
        super().__init__()
        ##原有的model参数
        self.context_length = context_length
        self.dropout = dropout
        if isinstance(vision_layers, (tuple, list)):
            print("HIHIHIHIHIHIHIHIHIHI")
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        # self.text_projection = nn.Linear(transformer_width, embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.audio_model = audio_model
        self.initialize_parameters()
        
        #自己模型的
        # 基础配置
        self.dropout = args.dropout #0.3
        # 创新点1，模态缺失
        self.miss_modal = args.miss_modal #none表示不缺失，audio表示缺失音频，video表示缺失视频
        self.miss_ratio = args.miss_ratio #0.5
        self.miss_noise = args.miss_noise #gaussian表示高斯噪声，uniform表示均匀噪声，random表示随机噪声，zero表示零噪声
        self.use_modal_share = args.use_modal_share #是否使用模态补全
        self.num_modal_class = args.num_modal_class #模态补全类别数，2
        # 创新点2，vae
        self.use_vae = args.use_vae #是否使用VAE
        self.vae_type = args.vae_type #CDSVAE, TransVAE, S3VAE
        self.f_dim = args.f_dim #256
        self.z_dim = args.z_dim #32
        self.g_dim = args.g_dim #128
        self.rnn_size = args.rnn_size #256
        self.f_rnn_layers = args.f_rnn_layers #1
        self.num_frames = args.num_frames #20
        self.final_dim = args.final_dim #512
        self.hidden_dim = args.hidden_dim #256
        if self.use_vae:
            if self.vae_type == 'TransVAE':
                self.vae = TransVAE(opt=args)
            if self.vae_type == 'CDSVAE':
                self.vae = CDSVAE(device="cuda", args=args)
        # 创新点3，反事实
        ##3.1 共性知识抽取
        self.use_knowledge = args.use_knowledge
        self.top_k = args.top_k #5
        self.use_static_knowledge = args.use_static_knowledge
        self.use_dynamic_knowledge = args.use_dynamic_knowledge
        self.use_video_knowledge = args.use_video_knowledge
        self.use_audio_knowledge = args.use_audio_knowledge
        self.use_image_knowledge = args.use_image_knowledge
        ## 3.2 反事实学习
        self.use_counterfactual = args.use_counterfactual
        self.intervented_type = args.intervened_type
        # 创新点4，梯度调整
        self.use_ogm = args.use_ogm
        self.args = args
        
        # 配置模型
        
        ##基础配置，编码器
        
        self.audio_encoder = nn.Linear(1024, self.hidden_dim)
        self.video_encoder = nn.Linear(512,self.hidden_dim)
        self.image_encoder = nn.Linear(512,self.hidden_dim)
        
        self.comb_layer = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(256+256+256+256, self.final_dim),#把问题和中间向量拼接
            nn.LayerNorm(self.final_dim),
            nn.ReLU(inplace=True),
        )
        self.comb_layer_intervened = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(256+256+256+256, self.final_dim),#把问题和中间向量拼接
            nn.LayerNorm(self.final_dim),
            nn.ReLU(inplace=True),
        )
        # self.comb_layer_withoutVAE = nn.Sequential(
        #     nn.Dropout(self.dropout),
        #     nn.Linear(512+self.num_frames*128, self.final_dim),#把问题和中间向量拼接
        #     nn.LayerNorm(self.mid_dim),
        #     nn.ReLU(inplace=True),
        # )
        # self.comb_layer_withoutVAE_knowledge = nn.Sequential(
        #     nn.Dropout(self.dropout),
        #     nn.Linear(512+256, self.mid_dim),#把问题和中间向量拼接
        #     nn.LayerNorm(self.mid_dim),
        #     nn.ReLU(inplace=True),
        # )
        ##创新点1，模态缺失
        if self.use_modal_share:
            self.comb_shaspec_layer = nn.Linear(256+256, 256)
            self.comb_shaspec_audio_layer = nn.Linear(256+256, 256)
            self.video_spec_feature = nn.Linear(256, 256)
            self.video_share_feature = nn.Linear(256, 256)
            self.audio_spec_feature = nn.Linear(256, 256)
            self.audio_share_feature = nn.Linear(256, 256)
            self.modal_classifier = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.final_dim*2+512, self.final_dim),#把问题和中间向量拼接
                nn.LayerNorm(self.final_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.final_dim, self.num_modal_class)#输出最终的结果，只是一个数字，拿这个数字去做最后的判别
            )
        
        ##创新点2，vae
        
        ##创新点3，反事实学习
        self.v_to_prior_dis_audio = nn.Linear(self.hidden_dim,self.hidden_dim)#音频信息去向矩阵A的接口
        self.v_to_prior_dis_static = nn.Linear(self.hidden_dim,self.hidden_dim)#静态信息去向矩阵A的接口
        self.v_to_prior_dis_dynamic = nn.Linear(self.hidden_dim,self.hidden_dim)#动态信息去向矩阵A的接口
        self.v_to_prior_dis_video = nn.Linear(self.num_frames*128,self.hidden_dim)#视频信息去向矩阵A的接口
        self.v_to_prior_dis_image = nn.Linear(self.hidden_dim,self.hidden_dim)#图像信息去向矩阵A的接口
        self.tao_video = 0.1
        ##创新点4，梯度调整
        self.use_ogm = args.use_ogm
        
        #共享信息编码器
        self.comb_shaspec_layer = nn.Linear(256+256, 256)
        self.comb_shaspec_audio_layer = nn.Linear(256+256, 256)
        self.spec_staic_feature = nn.Linear(256, 256)
        self.spec_dynamic_feature = nn.Linear(256, 256)
        self.spec_audio_feature = nn.Linear(256, 256)
        self.sha_static_audio_feature = nn.Linear(256, 256)
        self.sha_dynamic_audio_feature = nn.Linear(256, 256)
        self.modal_cls = nn.Linear(256, 2)
        
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
    
    # TODO: add attention mask based on bounding boxes
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype
    
    def encode_image(self, image):
        x = self.visual(image.type(self.dtype))
        # print("Image embedding")
        # print(x.shape)

        return x
    def encode_video(self, videos):
        batch, num_frames, _, _, _ = videos.shape
        videos = videos.reshape(videos.shape[0]*videos.shape[1], videos.shape[2], videos.shape[3], videos.shape[4])
        videos_feature = self.visual(videos.type(self.dtype))
        videos_feature = videos_feature.reshape(batch, num_frames, -1)
        # print("Video embedding")
        # print(videos[0].shape)
        return videos_feature
    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        # print(x.shape)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        # print(x.shape)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] 

        x = x @ self.text_projection
        # x = self.text_projection(x)

        return x
    def caluate_knowledge_relation_features(self, feature_dict):
        '''
        根据输入的特征计算其共性知识矩阵并返回经过共性知识矩阵处理过的矩阵和特征
        self里带有各种知识是否使用共性的指示,feature_dict中带有各种特征
        '''
        
        batch_size = feature_dict['video1_feature'].size(0)
        #将映射到共性知识空间和计算共性知识矩阵的过程分开
        #本函数主要是根据self的指示调用函数计算
        if self.use_static_knowledge:
            #计算静态知识矩阵
            v1_static_feature = feature_dict['v1_static_feature']
            v2_static_feature = feature_dict['v2_static_feature']
            all_features = torch.cat([v1_static_feature,v2_static_feature])#[batch_size*2, 256]
            all_features = self.v_to_prior_dis_static(all_features)
            v1_static_feature, v2_static_feature = knowledge_caluate(self, all_features, batch_size, use_intervened=False)
            feature_dict['v1_static_feature'] = v1_static_feature
            feature_dict['v2_static_feature'] = v2_static_feature
            if self.use_counterfactual:
                v1_static_feature_intervened, v2_static_feature_intervened = knowledge_caluate(self, all_features, batch_size, use_intervened=True)
                feature_dict['v1_static_feature_intervened'] = v1_static_feature_intervened
                feature_dict['v2_static_feature_intervened'] = v2_static_feature_intervened
        if self.use_dynamic_knowledge:
            
            v1_dynamic_feature = feature_dict['v1_dynamic_feature']
            v2_dynamic_feature = feature_dict['v2_dynamic_feature']
            all_features = torch.cat([v1_dynamic_feature,v2_dynamic_feature])
            all_features = self.v_to_prior_dis_dynamic(all_features)
            v1_dynamic_feature, v2_dynamic_feature = knowledge_caluate(self, all_features, batch_size, use_intervened=False)
            feature_dict['v1_dynamic_feature'] = v1_dynamic_feature
            feature_dict['v2_dynamic_feature'] = v2_dynamic_feature
            if self.use_counterfactual:
                v1_dynamic_feature_intervened, v2_dynamic_feature_intervened = knowledge_caluate(self, all_features, batch_size, use_intervened=True)
                feature_dict['v1_dynamic_feature_intervened'] = v1_dynamic_feature_intervened
                feature_dict['v2_dynamic_feature_intervened'] = v2_dynamic_feature_intervened
        if self.use_audio_knowledge:
            audio1_feature = feature_dict['audio1_features']
            audio2_feature = feature_dict['audio2_features']
            all_features = torch.cat([audio1_feature,audio2_feature])
            all_features = self.v_to_prior_dis_audio(all_features)
            audio1_feature, audio2_feature = knowledge_caluate(self, all_features, batch_size, use_intervened=False)
            feature_dict['audio1_feature'] = audio1_feature
            feature_dict['audio2_feature'] = audio2_feature
            if self.use_counterfactual:
                audio1_feature_intervened, audio2_feature_intervened = knowledge_caluate(self, all_features, batch_size, use_intervened=True)
                feature_dict['audio1_feature_intervened'] = audio1_feature_intervened
                feature_dict['audio2_feature_intervened'] = audio2_feature_intervened
        if self.use_image_knowledge:
            image1_feature = feature_dict['image1_features']
            image2_feature = feature_dict['image2_features']
            all_features = torch.cat([image1_feature,image2_feature])
            all_features = self.v_to_prior_dis_image(all_features)
            image1_feature, image2_feature = knowledge_caluate(self, all_features, batch_size, use_intervened=False)
            feature_dict['image1_feature'] = image1_feature
            feature_dict['image2_feature'] = image2_feature
            if self.use_counterfactual:
                image1_feature_intervened, image2_feature_intervened = knowledge_caluate(self, all_features, batch_size, use_intervened=True)
                feature_dict['image1_feature_intervened'] = image1_feature_intervened
                feature_dict['image2_feature_intervened'] = image2_feature_intervened
        if self.use_video_knowledge:
            video1_feature = feature_dict['video1_features']
            video2_feature = feature_dict['video2_features']
            all_features = torch.cat([video1_feature,video2_feature])
            all_features = self.v_to_prior_dis_video(all_features)
            video1_feature, video2_feature = knowledge_caluate(self, all_features, batch_size, use_intervened=False)
            feature_dict['video1_feature'] = video1_feature
            feature_dict['video2_feature'] = video2_feature
            if self.use_counterfactual:
                video1_feature_intervened, video2_feature_intervened = knowledge_caluate(self, all_features, batch_size, use_intervened=True)
                feature_dict['video1_feature_intervened'] = video1_feature_intervened
                feature_dict['video2_feature_intervened'] = video2_feature_intervened
        return feature_dict
    def forward(self, img1, audio1, img2, audio2, video1, video2,text, data_size,args, train_type):
        loss_dict = {}
        features_dict = dict()
        batch_size = img1.size(0)
        #计算抽取特征所需要的时间
        star_time = time.time()
        #抽取特征+编码到相同的维度
        img1_feature = self.encode_image(img1)
        img2_feature = self.encode_image(img2)
        img1_feature = self.image_encoder(img1_feature)#hidden_dim = 256
        img2_feature = self.image_encoder(img2_feature)
        audio1_feature = self.audio_model(audio1)
        audio2_feature = self.audio_model(audio2)
        audio1_feature = self.audio_encoder(audio1_feature)#hidden_dim = 256
        audio2_feature = self.audio_encoder(audio2_feature)
        video1_feature = self.encode_video(video1)
        video2_feature = self.encode_video(video2)
        video1_feature = self.video_encoder(video1_feature)#hidden_dim = 256
        video2_feature = self.video_encoder(video2_feature)
        text_feature = self.encode_text(text)# text_dim = 512
        
        
        img1_feature = img1_feature / img1_feature.norm(dim=-1, keepdim=True)
        img2_feature = img2_feature / img2_feature.norm(dim=-1, keepdim=True)
        audio1_feature = audio1_feature / audio1_feature.norm(dim=-1, keepdim=True)
        audio2_feature = audio2_feature / audio2_feature.norm(dim=-1, keepdim=True)
        video1_feature = video1_feature / video1_feature.norm(dim=-1, keepdim=True)
        video2_feature = video2_feature / video2_feature.norm(dim=-1, keepdim=True)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        end_time = time.time()
        # print("抽取特征所需要的时间为：",end_time-star_time)
        
        #feature_dict收纳
        features_dict['img1_feature'] = img1_feature
        features_dict['img2_feature'] = img2_feature
        features_dict['audio1_feature'] = audio1_feature
        features_dict['audio2_feature'] = audio2_feature
        features_dict['video1_feature'] = video1_feature
        features_dict['video2_feature'] = video2_feature
        features_dict['text_feature'] = text_feature

        # 第一步，模态缺失
        if self.miss_modal != "none":
            # 确定缺失位置
            index_limit = int(self.miss_ratio * batch_size)
            perm = torch.randperm(batch_size)
            random_indices = perm[:index_limit]
            non_zero_indices = perm[index_limit:]
            features_dict['random_indices'] = random_indices
            #确定缺失的模态和噪声类型，并得到缺失后的特征
            #video1_features_missing 和 video1_feature分别为缺失前的和缺失后的
            if self.miss_modal == "video":
                video1_feature_missing = video1_feature.clone()
                video2_feature_missing = video2_feature.clone()
                if self.miss_noise == "random":
                    video1_feature_missing[random_indices] = torch.randn(index_limit, self.hidden_dim).cuda()
                    video2_feature_missing[random_indices] = torch.randn(index_limit, self.hidden_dim).cuda()
                elif self.miss_noise == "uniform":
                    video1_feature_missing[random_indices] = torch.rand(index_limit, self.hidden_dim).cuda()
                    video2_feature_missing[random_indices] = torch.rand(index_limit, self.hidden_dim).cuda()
                elif self.miss_noise == "zero":
                    video1_feature_missing[random_indices] = 0
                    video2_feature_missing[random_indices] = 0
                features_dict['video1_feature_missing'] = video1_feature_missing
                features_dict['video2_feature_missing'] = video2_feature_missing
            if self.miss_modal == "audio":
                audio1_feature_missing = audio1_feature.clone()
                audio2_feature_missing = audio2_feature.clone()
                if self.miss_noise == "random":
                    audio1_feature_missing[random_indices] = torch.randn(index_limit, self.hidden_dim).cuda()
                    audio2_feature_missing[random_indices] = torch.randn(index_limit, self.hidden_dim).cuda()
                elif self.miss_noise == "uniform":
                    audio1_feature_missing[random_indices] = torch.rand(index_limit, self.hidden_dim).cuda()
                    audio2_feature_missing[random_indices] = torch.rand(index_limit, self.hidden_dim).cuda()
                elif self.miss_noise == "zero":
                    audio1_feature_missing[random_indices] = 0
                    audio2_feature_missing[random_indices] = 0
                features_dict['audio1_feature_missing'] = audio1_feature_missing
                features_dict['audio2_feature_missing'] = audio2_feature_missing
            #使用模态补全策略    
            if self.use_modal_share:
                #首先将video的特征的第1个维度平均
                video1_feature_missing = torch.mean(video1_feature_missing, dim=1)
                video2_feature_missing = torch.mean(video2_feature_missing, dim=1)
                
                obj1_spec_video_feature = self.video_spec_feature(video1_feature_missing)
                obj1_share_video_feature = self.video_share_feature(video1_feature_missing)
                obj2_spec_video_feature = self.video_spec_feature(video2_feature_missing)
                obj2_share_video_feature = self.video_share_feature(video2_feature_missing)
                
                obj1_spec_video_feature = self.video_spec_feature(video1_feature)
                obj1_share_video_feature = self.video_share_feature(video1_feature)
                obj2_spec_video_feature = self.video_spec_feature(video2_feature)
                obj2_share_video_feature = self.video_share_feature(video2_feature)
                
                obj1_spec_audio_feature = self.audio_spec_feature(audio1_feature)
                obj1_share_audio_feature = self.audio_share_feature(audio1_feature)
                obj2_spec_audio_feature = self.audio_spec_feature(audio2_feature)
                obj2_share_audio_feature = self.audio_share_feature(audio2_feature)
                
                if self.miss_modal == "audio":
                    #使用共享编码器的信息补全
                    obj1_share_audio_feature[random_indices] = obj1_share_video_feature[random_indices]
                    obj2_share_audio_feature[random_indices] = obj2_share_video_feature[random_indices]
                    
                    obj1_share_audio_feature = obj1_share_audio_feature / obj1_share_audio_feature.norm(dim=-1, keepdim=True)
                    obj2_share_audio_feature = obj2_share_audio_feature / obj2_share_audio_feature.norm(dim=-1, keepdim=True)
                    
                    #融合特征,添加了残差链接，融合后的特征加上了共享特征，详情见Multi-modal Learning with Missing Modality via Shared-Specific Feature Modelling， 公式2
                    audio1_feature = self.comb_shaspec_audio_layer(torch.cat([obj1_spec_audio_feature, obj1_share_audio_feature], dim=1)) + obj1_share_audio_feature
                    audio2_feature = self.comb_shaspec_audio_layer(torch.cat([obj2_spec_audio_feature, obj2_share_audio_feature], dim=1)) + obj2_share_audio_feature
                elif self.miss_modal == "video": 
                    #使用共享编码器的信息补全
                    obj1_share_video_feature[random_indices] = obj1_share_audio_feature[random_indices]
                    obj2_share_video_feature[random_indices] = obj2_share_audio_feature[random_indices]
                    
                    obj1_share_video_feature = obj1_share_video_feature / obj1_share_video_feature.norm(dim=-1, keepdim=True)
                    obj2_share_video_feature = obj2_share_video_feature / obj2_share_video_feature.norm(dim=-1, keepdim=True)
                    
                    video1_feature = self.comb_shaspec_layer(torch.cat([obj1_spec_video_feature, obj1_share_video_feature], dim=1)) + obj1_share_video_feature
                    video2_feature = self.comb_shaspec_layer(torch.cat([obj2_spec_video_feature, obj2_share_video_feature], dim=1)) + obj2_share_video_feature
                features_dict['audio1_feature'] = audio1_feature
                features_dict['audio2_feature'] = audio2_feature
                features_dict['video1_feature'] = video1_feature
                features_dict['video2_feature'] = video2_feature
                #目标第一，分类模态
                #0表示音频，1表示视频 [batch_size, 1]
                video_label = torch.ones([audio1_feature.size(0)])
                audio_label = torch.zeros([audio1_feature.size(0)])
                #设计损失函数
                loss_fn_modal_cls = torch.nn.CrossEntropyLoss()
                loss_fn_modal_share = torch.nn.L1Loss(reduction='mean')
                #计算损失
                obj1_spec_video_logits = self.modal_classifier(obj1_spec_video_feature)
                obj1_spec_audio_logits = self.modal_classifier(obj1_spec_audio_feature)
                obj2_spec_video_logits = self.modal_classifier(obj2_spec_video_feature)
                obj2_spec_audio_logits = self.modal_classifier(obj2_spec_audio_feature)
                obj1_spec_video_loss = loss_fn_modal_cls(obj1_spec_video_logits, video_label.long().cuda())#分类obj1视频模态
                obj1_spec_audio_loss = loss_fn_modal_cls(obj1_spec_audio_logits, audio_label.long().cuda())#分类obj1音频模态
                obj2_spec_video_loss = loss_fn_modal_cls(obj2_spec_video_logits, video_label.long().cuda())#分类obj2视频模态
                obj2_spec_audio_loss = loss_fn_modal_cls(obj2_spec_audio_logits, audio_label.long().cuda())#分类obj2音频模态
                #计算准确率
                obj1_spec_video_acc = (torch.argmax(obj1_spec_video_feature, dim=1) == video_label.long().cuda()).sum().item() / video_label.size(0)#计算obj1视频模态的准确率
                obj1_spec_audio_acc = (torch.argmax(obj1_spec_audio_feature, dim=1) == audio_label.long().cuda()).sum().item() / audio_label.size(0)#计算obj1音频模态的准确率
                obj2_spec_video_acc = (torch.argmax(obj2_spec_video_feature, dim=1) == video_label.long().cuda()).sum().item() / video_label.size(0)#计算obj2视频模态的准确率
                obj2_spec_audio_acc = (torch.argmax(obj2_spec_audio_feature, dim=1) == audio_label.long().cuda()).sum().item() / audio_label.size(0)#计算obj2音频模态的准确率
                
                wandb.log({train_type + "/obj1_spec_video_acc": obj1_spec_video_acc, train_type + "/obj1_spec_audio_acc": obj1_spec_audio_acc, train_type + "/obj2_spec_video_acc": obj2_spec_video_acc, train_type + "/obj2_spec_audio_acc": obj2_spec_audio_acc})
                
                 #目标第二，计算共享信息的L1损失
                obj1_share_loss = loss_fn_modal_share(obj1_share_video_feature, obj1_share_audio_feature)
                obj2_share_loss = loss_fn_modal_share(obj2_share_video_feature, obj2_share_audio_feature)
                wandb.log({train_type + "/obj1_share_loss": obj1_share_loss, train_type + "/obj2_share_loss": obj2_share_loss})
                
                loss_dict['obj1_spec_video_loss'] = obj1_spec_video_loss
                loss_dict['obj1_spec_audio_loss'] = obj1_spec_audio_loss
                loss_dict['obj2_spec_video_loss'] = obj2_spec_video_loss
                loss_dict['obj2_spec_audio_loss'] = obj2_spec_audio_loss
                loss_dict['obj1_share_loss'] = obj1_share_loss
                loss_dict['obj2_share_loss'] = obj2_share_loss
        #TODO add disentangled module
        # 第二步，vae
        if self.use_vae:
            if self.vae_type == 'CDSVAE':
                loss_1_vae, v1_static_feature, v1_dynamic_feature = compute_CDSVAE(self,video1, video1_feature, 'cuda', data_size)
                loss_2_vae, v2_static_feature, v2_dynamic_feature = compute_CDSVAE(self,video2, video2_feature, 'cuda', data_size)
            elif self.vae_type == 'TransVAE':
                o1_Recon_loss, o1_kld_f , o1_kld_z, o1_MI, loss_1_vae, v1_static_feature, _ ,v1_dynamic_feature = compute_TransVAE(self,video1, video1_feature, 'cuda', data_size)
                o2_Recon_loss, o2_kld_f , o2_kld_z, o2_MI, loss_2_vae, v2_static_feature, _, v2_dynamic_feature = compute_TransVAE(self,video2, video2_feature, 'cuda', data_size)
            elif self.vae_type == 'S3VAE':
                # TODO 等待补充
                pass
            loss_dict['loss_1_VAE'] = loss_1_vae
            loss_dict['loss_2_VAE'] = loss_2_vae
            loss_dict['o1_Recon_loss'] = o1_Recon_loss
            loss_dict['o1_kld_f'] = o1_kld_f
            loss_dict['o1_kld_z'] = o1_kld_z
            loss_dict['o1_MI'] = o1_MI
            loss_dict['o2_Recon_loss'] = o2_Recon_loss
            loss_dict['o2_kld_f'] = o2_kld_f
            loss_dict['o2_kld_z'] = o2_kld_z
            loss_dict['o2_MI'] = o2_MI
            features_dict['v1_static_feature'] = v1_static_feature
            features_dict['v1_dynamic_feature'] = v1_dynamic_feature
            features_dict['v2_static_feature'] = v2_static_feature
            features_dict['v2_dynamic_feature'] = v2_dynamic_feature
        # 第三步，共性知识抽取 + 反事实学习
        if self.use_knowledge:
            #将这个共性知识抽取模块进行包装
            features_dict = self.caluate_knowledge_relation_features(features_dict)
            #计算反事实学习的特征
        if self.use_counterfactual:
            #将不同的特征融合到一起
            obj1_feature = self.comb_layer(torch.cat([features_dict['img1_feature'],features_dict['audio1_feature'],features_dict['v1_static_feature'],features_dict['v1_dynamic_feature']], dim=1))
            obj2_feature = self.comb_layer(torch.cat([features_dict['img2_feature'],features_dict['audio2_feature'],features_dict['v2_static_feature'],features_dict['v2_dynamic_feature']], dim=1))
        features_dict['obj1_feature'] = obj1_feature
        features_dict['obj2_feature'] = obj2_feature
        # 第四步，梯度调整
        if self.use_ogm:
            # TODO 等待补充
            pass
        # 第五步,输出
        cs1 = nn.CosineSimilarity()(features_dict['text_feature'], obj1_feature)
        cs2 = nn.CosineSimilarity()(features_dict['text_feature'], obj2_feature)
        output = dict()
        output['cs1'] = cs1
        output['cs2'] = cs2
        return output, loss_dict, features_dict