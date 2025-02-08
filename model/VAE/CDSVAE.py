from collections import OrderedDict
from typing import Tuple, Union
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math

class LinearUnit(nn.Module):
    def __init__(self, in_features, out_features, batchnorm=True, nonlinearity=nn.LeakyReLU(0.2)):
        super(LinearUnit, self).__init__()
        if batchnorm is True:
            self.model = nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.BatchNorm1d(out_features), nonlinearity)
        else:
            self.model = nn.Sequential(
                    nn.Linear(in_features, out_features), nonlinearity)

    def forward(self, x):
        return self.model(x)

class CDSVAE(nn.Module):
    #VAE模型所代表的是特征抽取过程，所以应该引入特征抽取器
    def __init__(self, aud_feats=768,  img_feats=512,
                 f_dim = 256,z_dim = 32, g_dim = 128, rnn_size = 256, f_rnn_layers = 1,frames = 20, device = None,args = None):
        super(CDSVAE, self).__init__()
        self.img_feats = img_feats
        self.aud_feats = aud_feats
        self.f_dim = f_dim  # content 256 代表静态特征
        self.z_dim = z_dim  # motion 32 动态特征
        self.g_dim = g_dim  # frame feature 128 代表每一帧的特征
        # self.channels = channels  # frame feature 3
        self.hidden_dim = rnn_size #一层LSTM，256维
        self.f_rnn_layers = f_rnn_layers
        self.frames = frames #8帧视频，我们的视频居然要250帧？有点太多了，所以要抽

        self.video_encoder = nn.Linear(256, self.g_dim)
        self.video_encoder_2 = nn.Linear(256, 768)
        self.device = device
        self.video_decoder = nn.Linear(288, 768)
        # Frame encoder and decoder 不需要encoder
        # self.encoder = encoder(self.g_dim, self.channels)#得到对视频的特征抽取 
        # self.decoder = decoder(self.z_dim + self.f_dim, self.channels) #解耦静态和动态特征，根据其进行图像的重建
        #直接使用ViT抽取特征
        # Prior of content is a uniform Gaussian and prior of the dynamics is an LSTM
        self.z_prior_lstm_ly1 = nn.LSTMCell(self.z_dim, self.hidden_dim) #构建一个LSTM
        self.z_prior_lstm_ly2 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)#构建一个LSTM单元

        self.z_prior_mean = nn.Linear(self.hidden_dim, self.z_dim) #将动态特征取其均值
        self.z_prior_logvar = nn.Linear(self.hidden_dim, self.z_dim)#取动态特征的方差，都是32维的

        self.z_lstm = nn.LSTM(self.g_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)#双向LSTM计算content
        self.f_mean = LinearUnit(self.hidden_dim * 2, self.f_dim, False)#将其映射到静态信息的维度
        self.f_logvar = LinearUnit(self.hidden_dim * 2, self.f_dim, False)#这个也是

        self.z_rnn = nn.RNN(self.hidden_dim * 2, self.hidden_dim, batch_first=True)#使用RNN的原因？
        # Each timestep is for each z so no reshaping and feature mixing
        self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)#两个z？一个先验，一个后验？
        self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)

    def encode_and_sample_post(self, x, x_features):
        #对拿到的x进行Vit抽取特征
        #conv_x_output是视频刚抽取出来的特征，而conv_x是LSTM所使用的特征
        conv_x,conv_x_output = self.encoder_frame(x_features) #from [32,8,224,224] -> [32,8,512]
        # conv_x :[32,8,128], conv_x_output :[32,8,128]
        # pass the bidirectional lstm，经过双向LSTM，这里的conv_x是[batchsize, frame, embedding]
        lstm_out, _ = self.z_lstm(conv_x) #得到了[32,8,256]
        # get f:得到静态向量
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]#第一个时间步和最后一个时间步合起来得到近似后验
        frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]#取最后一帧的特征XT-1
        lstm_out_f = torch.cat((frontal, backward), dim=1)#将这两个合并在一起[32,512]
        f_mean = self.f_mean(lstm_out_f)#拿到f
        f_logvar = self.f_logvar(lstm_out_f)
        f_post = self.reparameterize(f_mean, f_logvar, random_sampling=True)#重参数化 他是只针对一帧的

        # pass to one direction rnn
        features, _ = self.z_rnn(lstm_out)#把8帧全部输入其中，得到最后的结果
        z_mean = self.z_mean(features)
        z_logvar = self.z_logvar(features)
        z_post = self.reparameterize(z_mean, z_logvar, random_sampling=True)#对于动态信息重参数化


        # f_mean is list if triple else not
        return f_mean, f_logvar, f_post, z_mean, z_logvar, z_post,conv_x_output

    def forward(self, x, x_features):
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post,conv_x = self.encode_and_sample_post(x, x_features)#输入的x本身就一个视频，带帧的
        z_mean_prior, z_logvar_prior, z_prior = self.sample_z_prior_train(z_post, random_sampling=self.training)

        z_flatten = z_post.view(-1, z_post.shape[2])


        f_expand = f_post.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_post, f_expand), dim=2)
        recon_x = self.video_decoder(zf)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, z_prior, \
               recon_x,conv_x

    def forward_fixed_motion(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        z_repeat = z_post[0].repeat(z_post.shape[0], 1, 1)
        f_expand = f_post.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_repeat, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, recon_x


    def forward_fixed_content(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        f_repeat = f_post[0].repeat(f_post.shape[0], 1)
        f_expand = f_repeat.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        zf = torch.cat((z_post, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, recon_x

    def forward_fixed_content_for_classification(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=True)
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        f_expand = f_mean.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        zf = torch.cat((z_mean_prior, f_expand), dim=2)
        recon_x_sample = self.decoder(zf)


        zf = torch.cat((z_mean_post, f_expand), dim=2)
        recon_x = self.decoder(zf)

        return recon_x_sample, recon_x

    def forward_fixed_motion_for_classification(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=True)
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        f_prior = self.reparameterize(torch.zeros(f_mean.shape).cuda(), torch.zeros(f_logvar.shape).cuda(),
                                        random_sampling=True)
        f_expand = f_prior.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_mean_post, f_expand), dim=2)
        recon_x_sample = self.decoder(zf)

        f_expand = f_mean.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_mean_post, f_expand), dim=2)
        recon_x = self.decoder(zf)

        return recon_x_sample, recon_x

    def encoder_frame(self, x_features):
        # input x is the features of a list of length Frames [batchsize,num_frames,512]
        # convert it to [batchsize, frames, channels, size, size]
        # x = torch.stack(x, dim=1)
        # [batch_size, frames, channels, size, size] to [batch_size * frames, channels, size, size]
        # x_shape = x.shape #[32,8,3,224,224]
        # x = x.view(-1, x_shape[-3], x_shape[-2], x_shape[-1]) #[32*8,3, 224,224]
        # x_embed_output = self.image_model(x).pooler_output#[32*8,768]
        batch_size, num_frames, _ = x_features.shape
        x = x_features.reshape(-1,256)
        x_embed = self.video_encoder(x_features)#[batch_size*num_frames, 128]
        x_features = self.video_encoder_2(x_features)
        return x_embed.view(batch_size, num_frames, -1), x_features #返回的是[batchsize, frame, embedding][32,8,128]

    def reparameterize(self, mean, logvar, random_sampling=True):
        # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5 * logvar)
            z = mean + eps * std
            return z
        else:
            return mean

    def sample_z_prior_test(self, n_sample, n_frame, random_sampling=True):
        z_out = None  # This will ultimately store all z_s in the format [batch_size, frames, z_dim]
        z_means = None
        z_logvars = None
        batch_size = n_sample

        z_t = torch.zeros(batch_size, self.z_dim).cuda()
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()

        for i in range(n_frame):
            # two layer LSTM and two one-layer FC
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(z_t, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))

            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            z_prior = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
                z_out = z_prior.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                # If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
                z_out = torch.cat((z_out, z_prior.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
                # z_t = z_post[:,i,:]
            z_t = z_prior
        return z_means, z_logvars, z_out

    def sample_z_prior_train(self, z_post, random_sampling=True):
        z_out = None  # This will ultimately store all z_s in the format [batch_size, frames, z_dim]
        z_means = None
        z_logvars = None
        batch_size = z_post.shape[0]

        z_t = torch.zeros(batch_size, self.z_dim, dtype=torch.float32).to(self.device)
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim,dtype=torch.float32).to(self.device)
        c_t_ly1 = torch.zeros(batch_size, self.hidden_dim,dtype=torch.float32).to(self.device)
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim,dtype=torch.float32).to(self.device)
        c_t_ly2 = torch.zeros(batch_size, self.hidden_dim,dtype=torch.float32).to(self.device)

        for i in range(self.frames):
            # two layer LSTM and two one-layer FC
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(z_t, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))

            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            z_prior = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
                z_out = z_prior.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                # If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
                z_out = torch.cat((z_out, z_prior.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
            z_t = z_post[:,i,:]
        return z_means, z_logvars, z_out

    # If random sampling is true, reparametrization occurs else z_t is just set to the mean
    def sample_z(self, batch_size, random_sampling=True):
        z_out = None  # This will ultimately store all z_s in the format [batch_size, frames, z_dim]
        z_means = None
        z_logvars = None

        # All states are initially set to 0, especially z_0 = 0
        z_t = torch.zeros(batch_size, self.z_dim).cuda()
        # z_mean_t = torch.zeros(batch_size, self.z_dim)
        # z_logvar_t = torch.zeros(batch_size, self.z_dim)
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        for _ in range(self.frames):
            # h_t, c_t = self.z_prior_lstm(z_t, (h_t, c_t))
            # two layer LSTM and two one-layer FC
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(z_t, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))

            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            z_t = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
                z_out = z_t.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                # If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
                z_out = torch.cat((z_out, z_t.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
        return z_means, z_logvars, z_out