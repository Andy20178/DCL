import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torch.autograd import Function


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, beta):
        ctx.beta = beta
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.beta
        return grad_input, None    
    

class TransVAE(nn.Module):
    def __init__(self, opt):
        super(TransVAE, self).__init__()
        self.f_dim = 256
        self.z_dim = 256 #dimensionality of z_t
        self.fc_dim = 512
        self.input_dim = 256
        self.frames = 8
        self.frame_aggregation = "rnn"
        self.dropout_rate = 0.5
        self.prior_sample = 'post'
        self.add_fc = 2
        self.f_rnn_layers = 1
        self.z_dim = 256 #dimensionality of z_t 
        self.enc_fc_layer1 = nn.Linear(self.input_dim, self.fc_dim)
        self.dec_fc_layer1 = nn.Linear(self.fc_dim, self.input_dim)
        self.fc_output_dim = self.fc_dim    
        
        self.bn_enc_layer1 = nn.BatchNorm1d(self.fc_output_dim)
        self.bn_dec_layer1 = nn.BatchNorm1d(self.input_dim)
        self.en_video_feat = nn.Linear(self.z_dim * 2, self.input_dim)
        if self.add_fc > 1:
            self.enc_fc_layer2 = nn.Linear(self.fc_dim, self.fc_dim)
            self.dec_fc_layer2 = nn.Linear(self.fc_dim, self.fc_dim)
            self.fc_output_dim = self.fc_dim
            self.bn_enc_layer2 = nn.BatchNorm1d(self.fc_output_dim)
            self.bn_dec_layer2 = nn.BatchNorm1d(self.fc_dim)
        self.bn_dec_layer3 = nn.BatchNorm1d(self.input_dim)
        self.z_2_out = nn.Linear(self.z_dim + self.f_dim, self.fc_output_dim)
        self.relu = nn.LeakyReLU(0.1)
        self.dropout_f = nn.Dropout(p=self.dropout_rate)
        self.dropout_v = nn.Dropout(p=self.dropout_rate)
        self.hidden_dim = self.z_dim
        self.f_rnn_layers = self.f_rnn_layers

        self.z_prior_lstm_ly1 = nn.LSTMCell(self.z_dim, self.hidden_dim)
        self.z_prior_lstm_ly2 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.z_prior_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_prior_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        self.z_lstm = nn.LSTM(self.fc_output_dim, self.hidden_dim, self.f_rnn_layers, bidirectional=True, batch_first=True)
        self.f_mean = nn.Linear(self.hidden_dim * 2, self.f_dim)
        self.f_logvar = nn.Linear(self.hidden_dim * 2, self.f_dim)

        self.z_rnn = nn.RNN(self.hidden_dim * 2, self.hidden_dim, batch_first=True)
        self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)


        self.bilstm = nn.LSTM(self.z_dim, self.z_dim * 2, self.f_rnn_layers, bidirectional=True, batch_first=True)
        self.feat_aggregated_dim = self.z_dim * 2
    
    def encode_and_sample_post(self, x):
        
        conv_x = self.encoder_frame(x)  # [batchsize, frames, 512]
        lstm_out, _ = self.z_lstm(conv_x)  # Input encoded into LSTM
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
        lstm_out_f = torch.cat((frontal, backward), dim=1)
        f_mean = self.f_mean(lstm_out_f)  # Obtained from the encoded results
        f_logvar = self.f_logvar(lstm_out_f)  # Obtained from the encoded results
        f_post = self.reparameterize(f_mean, f_logvar, random_sampling=False)  # Reparameterization to obtain the prior of static information
        features, _ = self.z_rnn(lstm_out)
        z_mean = self.z_mean(features)
        z_logvar = self.z_logvar(features)
        z_post = self.reparameterize(z_mean, z_logvar, random_sampling=False)  # Obtain the prior of dynamic information
        return f_mean, f_logvar, f_post, z_mean, z_logvar, z_post
    
    def decoder_frame(self,zf):
        zf = self.z_2_out(zf)
        zf = self.relu(zf)
        if self.add_fc > 2:
            zf = self.dec_fc_layer3(zf)
            zf = self.bn_dec_layer3(zf)
            zf = self.relu(zf)
        if self.add_fc > 1:
            zf = self.dec_fc_layer2(zf)
            num_frames = zf.size()[1]
            zf = zf.view(-1, self.fc_dim)
            zf = self.bn_dec_layer2(zf)
            zf = zf.view(-1, num_frames, self.fc_dim)
            zf = self.relu(zf)
        zf = self.dec_fc_layer1(zf)
        num_frames = zf.size()[1]
        zf = zf.view(-1, self.input_dim)
        zf = self.bn_dec_layer3(zf)
        zf = zf.view(-1, num_frames, self.input_dim)
        recon_x = self.relu(zf)
        return recon_x

    def encoder_frame(self, x):
        x_embed = self.enc_fc_layer1(x)
        num_frames = x_embed.size()[1]
        x_embed = self.bn_enc_layer1(x_embed.view(-1, self.fc_output_dim))
        x_embed = x_embed.view(-1, num_frames, self.fc_output_dim)
        x_embed = self.relu(x_embed)
        if self.add_fc > 1:
            x_embed = self.enc_fc_layer2(x_embed.view(-1, self.fc_output_dim))
            x_embed = self.bn_enc_layer2(x_embed)
            x_embed = x_embed.view(-1, num_frames, self.fc_output_dim)
            x_embed = self.relu(x_embed)
        return x_embed 
    
    def reparameterize(self, mean, logvar, random_sampling=True):
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5 * logvar)
            z = mean + eps * std
            return z
        else:
            return mean

    def sample_z_prior_train(self, z_post, random_sampling=True):
        z_out = None
        z_means = None
        z_logvars = None
        batch_size = z_post.shape[0]
        z_t = torch.zeros(batch_size, self.z_dim).cuda()
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        for i in range(self.frames):
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(z_t, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))
            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            z_prior = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                z_out = z_prior.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                z_out = torch.cat((z_out, z_prior.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
            z_t = z_post[:,i,:]
        return z_means, z_logvars, z_out

    def sample_z(self, batch_size, random_sampling=True):
        z_out = None
        z_means = None
        z_logvars = None
        z_t = torch.zeros(batch_size, self.z_dim).cuda()
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        for _ in range(self.frames):
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(z_t, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))
            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            z_t = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                z_out = z_t.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                z_out = torch.cat((z_out, z_t.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
        return z_means, z_logvars, z_out

    def forward(self, x):
        
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)
        if self.prior_sample == 'random':
            z_mean_prior, z_logvar_prior, z_prior = self.sample_z(z_post.size(0),random_sampling=False)
        elif self.prior_sample == 'post':
            z_mean_prior, z_logvar_prior, z_prior = self.sample_z_prior_train(z_post, random_sampling=False)
        f_expand = f_post.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_post, f_expand), dim=2)
        
        recon_x = self.decoder_frame(zf)
        
        self.bilstm.flatten_parameters()
        z_post_video_feat, _ = self.bilstm(z_post)
        backward = z_post_video_feat[:, 0, self.z_dim:2 * self.z_dim]
        frontal = z_post_video_feat[:, self.frames - 1, 0:self.z_dim]
        z_post_video_feat = torch.cat((frontal, backward), dim=1)
        z_post_video_feat = self.en_video_feat(z_post_video_feat)
        z_post_video_feat = self.dropout_v(z_post_video_feat)

        
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, z_prior, recon_x, z_post_video_feat
    
    