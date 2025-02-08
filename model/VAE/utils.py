from collections import OrderedDict
from typing import Tuple, Union
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math
import wandb
"""
VAE要用到的一些函数,主要是帮助计算Compute_TransVAE和Compute_CDSVAE的
"""
def matrix_log_density_gaussian(x, mu, logvar):
    batch_size, dim = x.shape
    x = x.view(batch_size, 1, dim)
    mu = mu.view(1, batch_size, dim)
    logvar = logvar.view(1, batch_size, dim)
    return log_density_gaussian(x, mu, logvar)

def log_density_gaussian(x, mu, logvar):
    normalization = - 0.5 * (math.log(2 * math.pi) + logvar)
    inv_var = torch.exp(-logvar)
    log_density = normalization - 0.5 * ((x - mu)**2 * inv_var)
    return log_density

def log_importance_weight_matrix(batch_size, dataset_size):
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
    W.view(-1)[::M + 1] = 1 / N
    W.view(-1)[1::M + 1] = strat_weight
    W[M - 1, 0] = strat_weight
    return W.log()

def compute_mi(latent_sample, latent_dist):
    log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_log_pz_qz_prodzi_qzCx(latent_sample,
                                                                         latent_dist,
                                                                         None,
                                                                         is_mss=False)
    # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
    mi_loss = (log_q_zCx - log_qz).mean()

    return mi_loss

def logsumexp(value, dim=None, keepdim=False):
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        raise ValueError('Must specify the dimension.')
    
def _get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, n_data, is_mss=True):
    batch_size, hidden_dim = latent_sample.shape
    #print("latent_sample:", latent_sample.shape)
    #print("latent_dist:", len(latent_dist), latent_dist[0].shape, latent_dist[1].shape)
    #print("is_mss:", is_mss)

    # calculate log q(z|x)
    log_q_zCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)

    # calculate log p(z)
    # mean and log var is 0
    #zeros = torch.zeros_like(latent_sample)
    #log_pz = log_density_gaussian(latent_sample, zeros, zeros).sum(1)

    mat_log_qz = matrix_log_density_gaussian(latent_sample, *latent_dist)

    if is_mss:
        # use stratification
        log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(latent_sample.device)
        mat_log_qz = mat_log_qz + log_iw_mat.view(batch_size, batch_size, 1)

    log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
    #log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False).sum(1)
    #return log_pz, log_qz, log_prod_qzi, log_q_zCx
    return None, log_qz, None, log_q_zCx

def log_density(sample, mu, logsigma):
    mu = mu.type_as(sample)
    logsigma = logsigma.type_as(sample)
    c = torch.Tensor([np.log(2 * np.pi)]).type_as(sample.data)

    inv_sigma = torch.exp(-logsigma)
    tmp = (sample - mu) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * logsigma + c)

def log_importance_weight_matrix(batch_size, dataset_size):
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
    W.view(-1)[::M+1] = 1 / N
    W.view(-1)[1::M+1] = strat_weight
    W[M-1, 0] = strat_weight
    return W.log()

def loss_fn_new(original_seq, recon_seq, 
                f_mean, f_logvar, 
                z_post_mean, z_post_logvar, z_post,
                z_prior_mean, z_prior_logvar, z_prior
    ):
    '''
    TransVAE的损失函数
    '''
    batch_size = original_seq.size(0)
    mse = F.mse_loss(recon_seq, original_seq, reduction='sum')/batch_size
    f_mean = f_mean.view((-1, f_mean.shape[-1]))
    f_logvar = f_logvar.view((-1, f_logvar.shape[-1]))
    
    kld_f = -0.5 * \
        torch.sum(1 + f_logvar - torch.pow(f_mean, 2) -
                  torch.exp(f_logvar))/batch_size

    z_post_var = torch.exp(z_post_logvar)
    z_prior_var = torch.exp(z_prior_logvar)
    kld_z = 0.5 * torch.sum(z_prior_logvar - z_post_logvar + ((z_post_var +
                            torch.pow(z_post_mean - z_prior_mean, 2)) / z_prior_var) - 1)/batch_size

    return {'mse': mse, 'kld_f': kld_f, 'kld_z': kld_z}

class contrastive_loss(nn.Module):
    '''
    给CDSVAE的对比损失
    '''
    def __init__(self, tau=1, normalize=False,device = None):
        super(contrastive_loss, self).__init__()
        self.tau = tau
        self.normalize = normalize
        self.device = device
    def forward(self, xi, xj):

        x = torch.cat((xi, xj), dim=0) # [128, 256]

        is_cuda = x.is_cuda
        sim_mat = torch.mm(x, x.T)
        if self.normalize: # False
            sim_mat_denom = torch.mm(torch.norm(x, dim=1).unsqueeze(1), torch.norm(x, dim=1).unsqueeze(1).T)
            sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)

        sim_mat = torch.exp(sim_mat / self.tau)

        # no diag because it's not diffrentiable -> sum - exp(1 / tau)
        # diag_ind = torch.eye(xi.size(0) * 2).bool()
        # diag_ind = diag_ind.cuda() if use_cuda else diag_ind

        # sim_mat = sim_mat.masked_fill_(diag_ind, 0)

        # top
        if self.normalize:
            sim_mat_denom = torch.norm(xi, dim=1) * torch.norm(xj, dim=1)
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / sim_mat_denom / self.tau)
        else:
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / self.tau)

        sim_match = torch.cat((sim_match, sim_match), dim=0)

        norm_sum = torch.exp(torch.ones(x.size(0)) / self.tau)
        norm_sum = norm_sum.to(self.device) if is_cuda else norm_sum
        loss = torch.mean(-torch.log(sim_match / (torch.sum(sim_mat, dim=-1) - norm_sum)))

        return loss

def compute_TransVAE(self,video,video_features,device, data_size):
        video_shape = video.shape#[32,8,3,224,224]
        batch_size, num_frames, channels, H, W = video_shape
        batch_size, num_frames, feature_dim = video_features.shape
        x = video_features
        f_mean, f_logvar, f_post, z_post_mean, z_post_logvar, z_post, z_prior_mean, z_prior_logvar, z_prior, recon_x, z_post_video_feat= self.vae(x)
        vae_loss_dict = loss_fn_new(x, recon_x, f_mean, f_logvar, z_post_mean, z_post_logvar, z_post, z_prior_mean, z_prior_logvar, z_prior)
        VAE_loss = vae_loss_dict['mse'] + 1*vae_loss_dict['kld_f'] + 1*vae_loss_dict['kld_z']
        loss = 1 * VAE_loss
        
        self.use_MI = True
        #(II)calculate the mutual infomation of f and z
        if self.use_MI:
            _logq_f_tmp = log_density(f_post.unsqueeze(0).repeat(num_frames, 1, 1).view(num_frames, batch_size, 1, 256),
                                      f_mean.unsqueeze(0).repeat(num_frames, 1, 1).view(num_frames, 1, batch_size, 256),
                                      f_logvar.unsqueeze(0).repeat(num_frames, 1, 1).view(num_frames, 1, batch_size, 256))

            # n_frame x batch_size x batch_size x f_dim
            _logq_z_tmp = log_density(z_post.transpose(0, 1).view(num_frames, batch_size, 1, 256),
                                      z_post_mean.transpose(0, 1).view(num_frames, 1, batch_size, 256),
                                      z_post_logvar.transpose(0, 1).view(num_frames, 1, batch_size, 256))
            _logq_fz_tmp = torch.cat((_logq_f_tmp, _logq_z_tmp), dim=3)

            logq_f = (logsumexp(_logq_f_tmp.sum(3), dim=2, keepdim=False) - math.log(batch_size * data_size))
            logq_z = (logsumexp(_logq_z_tmp.sum(3), dim=2, keepdim=False) - math.log(batch_size * data_size))
            logq_fz = (logsumexp(_logq_fz_tmp.sum(3), dim=2, keepdim=False) - math.log(batch_size * data_size))

            # n_frame x batch_size
            # some sample are wired negative
            loss_MI = F.relu(logq_fz - logq_f - logq_z).mean()
            loss += 50 * loss_MI
        return vae_loss_dict['mse'], vae_loss_dict['kld_f'],vae_loss_dict['kld_z'], 50*loss_MI, loss, f_post, z_post,z_post_video_feat 
def compute_CDSVAE(self,video,video_features,device, data_size):
        
        
        contras_fn = contrastive_loss(tau=0.5, normalize=True, device= device)
        
        # video = video[:,::5,:,:]
        num_frames = video.shape[1]#[batch, num_frames, channels, size, size]
        video_shape = video.shape#[32,8,3,224,224]
        batch_size, num_frames, channels, H, W = video_shape
        #构造静态数据增强，使用将第二维度的帧重新排列的方式
        random_indices = torch.randperm(num_frames)
        video_aug_static_features = video_features[:,random_indices,:]
        
        
        #构造动态数据增强，将224进行高斯模糊
        # blur_transform = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        # video_aug_dynamic = blur_transform(video.view(-1,3,video_shape[3],video_shape[4]))#[8,20,3,224,224]
        
        # video_aug_dynamic = video_aug_dynamic.view(batch_size,num_frames,channels,video_shape[3],video_shape[4])
        f_mean, f_logvar, f, z_post_mean, \
        z_post_logvar, z_post, z_prior_mean, z_prior_logvar, \
        z_prior,recon_x,conv_x_output = self.CDSVAE(video, video_features)
        
        f_mean_c, f_logvar_c, f_c, _, _, _, _, _, _, _, _ = self.CDSVAE(video_aug_static_features, video_features)#只看静态特征
        
        _, _, _, z_post_mean_m, z_post_logvar_m, z_post_m, _, _, _, _,_ = self.CDSVAE(video_features,video_features)#只看动态特征，可以注意到使用的是Z——post
        
        #需要将CDSVAE包装成针对单个obj的
        #对于video1计算，下面全部改成video1
        mi_xs = compute_mi(f, (f_mean, f_logvar))#计算互信息，这个过程我们就不要仔细了解了
        n_bs = z_post.shape[0] #batchsize 32
        
        mi_xzs = [compute_mi(z_post_t, (z_post_mean_t, z_post_logvar_t)) \
                    for z_post_t, z_post_mean_t, z_post_logvar_t in \
                    zip(z_post.permute(1,0,2), z_post_mean.permute(1,0,2), z_post_logvar.permute(1,0,2))]
        mi_xz = torch.stack(mi_xzs).sum()#互信息

        
        l_recon = F.mse_loss(recon_x, conv_x_output.view(n_bs,num_frames,-1), reduction='sum')*0.1#计算重建loss

        f_mean = f_mean.view((-1, f_mean.shape[-1])) # [128, 256]表示静态信息
        f_logvar = f_logvar.view((-1, f_logvar.shape[-1])) # [128, 256]
        kld_f = -0.5 * torch.sum(1 + f_logvar - torch.pow(f_mean,2) - torch.exp(f_logvar))

        z_post_var = torch.exp(z_post_logvar) # [128, 8, 32]
        z_prior_var = torch.exp(z_prior_logvar) # [128, 8, 32]
        kld_z = 0.5 * torch.sum(z_prior_logvar - z_post_logvar +
                                ((z_post_var + torch.pow(z_post_mean - z_prior_mean, 2)) / z_prior_var) - 1)

        l_recon, kld_f, kld_z = l_recon / n_bs, kld_f / n_bs, kld_z / n_bs

        batch_size, n_frame, z_dim = z_post_mean.size()#每一步的

        con_loss_c = contras_fn(f_mean, f_mean_c)#对比loss
        con_loss_m = contras_fn(z_post_mean.view(batch_size, -1), z_post_mean_m.view(batch_size, -1))#对比loss

        # calculate the mutual infomation of f and z
        mi_fz = torch.zeros((1)).cuda()
        if True: # 0.1
            # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
            # batch_size x batch_size x f_dim
            _logq_f_tmp = log_density(f.unsqueeze(0).repeat(n_frame, 1, 1).view(n_frame, batch_size, 1, self.f_dim), # [8, 128, 1, 256]
                                    f_mean.unsqueeze(0).repeat(n_frame, 1, 1).view(n_frame, 1, batch_size, self.f_dim), # [8, 1, 128, 256]
                                    f_logvar.unsqueeze(0).repeat(n_frame, 1, 1).view(n_frame, 1, batch_size, self.f_dim)) # [8, 1, 128, 256]

            # n_frame x batch_size x batch_size x f_dim
            _logq_z_tmp = log_density(z_post.transpose(0, 1).view(n_frame, batch_size, 1, z_dim), # [8, 128, 1, 32]
                                    z_post_mean.transpose(0, 1).view(n_frame, 1, batch_size, z_dim), # [8, 1, 128, 32]
                                    z_post_logvar.transpose(0, 1).view(n_frame, 1, batch_size, z_dim)) # [8, 1, 128, 32]

            _logq_fz_tmp = torch.cat((_logq_f_tmp, _logq_z_tmp), dim=3) # [8, 128, 128, 288]

            logq_f = (logsumexp(_logq_f_tmp.sum(3), dim=2, keepdim=False) - math.log(batch_size * data_size)) # [8, 128]
            logq_z = (logsumexp(_logq_z_tmp.sum(3), dim=2, keepdim=False) - math.log(batch_size * data_size)) # [8, 128]
            logq_fz = (logsumexp(_logq_fz_tmp.sum(3), dim=2, keepdim=False) - math.log(batch_size * data_size)) # [8, 128]
            # n_frame x batch_size
            mi_fz = F.relu(logq_fz - logq_f - logq_z).mean()

        loss = l_recon + kld_f*1 + kld_z*1 + mi_fz
        if self.training:
            wandb.log({"l_recon":l_recon,"kld_f":kld_f,"kld_z":kld_z,"mi_fz":mi_fz,"con_loss_c":con_loss_c,"con_loss_m":con_loss_m})
        #print("l_recon:",l_recon,"kld_f:",kld_f,"kld_z:",kld_z,"mi_fz:",mi_fz,"con_loss_c:",con_loss_c,"con_loss_m:",con_loss_m)
        loss += con_loss_c*10
        loss += con_loss_m*10
        return loss, f, z_post