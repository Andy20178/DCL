import torch
import numpy as np
import random
import torch.nn as nn
from model.audioclip_finetune import AudioCLIPFinetune
import argparse
from utils_extra import get_logger, load_model
from dataset.dataset import PACSImageAudioDataset
from torchvision import transforms
from transformers import AdamW
import torch.optim as optim
import os
from PIL import Image
import utils.transforms as audio_transforms
import wandb
import json
import librosa
import clip
from utils_extra import load_dataset, compute_acc_loss
from trainer import train_model
from tqdm import tqdm   #进度条
import pdb
'''
设置随机种子
'''
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)



'''
添加训练参数
'''
parser = argparse.ArgumentParser()
#路径情况
parser.add_argument('--data_dir', default="/data/lcs/PACS_data", type=str)
parser.add_argument('--save_path', default="results", type=str,help='用来保存全部的结果, 其中包括log, checkpoints, vis, 以及wandb')
parser.add_argument('--model_filename', default='AudioCLIP-Partial-Training.pt', type=str, help='预训练的AudioCLIP模型的路径')

# 数据输入参数
parser.add_argument('--use_audio', default=True, type=bool)
parser.add_argument('--use_video',default=True,type=bool)
parser.add_argument('--use_image',default=False,type=bool)

#模型选择
#创新点1，模态补全，modal_share
parser.add_argument('--miss_modal', default="none", type=str, help='选择缺失的模态种类, none表示不缺失, audio表示缺失音频, video表示缺失视频')
parser.add_argument('--miss_ratio', default=0.5, type=float, help='缺失率')
parser.add_argument('--miss_noise', default="gaussian", type=str, help='缺失的噪声类型, gaussian表示高斯噪声, \
                                    uniform表示均匀噪声,random表示随机噪声, zero表示零噪声')
parser.add_argument('--use_modal_share', default=True, type=bool, help='是否使用模态补全')
parser.add_argument('--num_modal_class', default=2, type=int, help='模态补全的类别数')
#创新点2，解耦VAE
parser.add_argument('--use_vae', default=True, type=bool, help='是否使用VAE')
parser.add_argument('--vae_type', default="TransVAE", type=str, help='VAE的类型, CDSVAE(NeurIPS2021), TransVAE(NeurIPS2023),\
                                    S3VAE(CVPR2020),这其中已经包含了不同的互信息计算方法')
parser.add_argument('--f_dim', default=256, type=int, help='特征维度')
parser.add_argument('--z_dim', default=32, type=int, help='z的维度')
parser.add_argument('--g_dim', default=128, type=int, help='g的维度,vae中每一帧的维度')
parser.add_argument('--rnn_size', default=256, type=int, help='rnn的size')
parser.add_argument('--f_rnn_layers', default=1, type=int, help='f_rnn的层数')
parser.add_argument('--num_frames', default=20, type=int, help='视频里抽取的帧数')
parser.add_argument('--final_dim', default=512, type=int, help='最后输出的维度')
parser.add_argument('--hidden_dim', default=256, type=int, help='隐藏维度，所有中间变量都是这个维度')

#创新点3，反事实学习
## 3.1 共性知识抽取
parser.add_argument('--use_knowledge', default=True, type=bool, help='是否使用共性知识')
parser.add_argument('--use_static_knowledge', default=True, type=bool, help='是否使用静态共性知识')
parser.add_argument('--use_dynamic_knowledge', default=True, type=bool, help='是否使用动态共性知识')
parser.add_argument('--use_audio_knowledge', default=False, type=bool, help='是否使用音频共性知识')
parser.add_argument('--use_image_knowledge', default=False, type=bool, help='是否使用图像共性知识')
parser.add_argument('--use_video_knowledge', default=False, type=bool, help='是否使用原始视频共性知识')
parser.add_argument('--top_k', default=5, type=int, help='共性知识的top_k')
## 3.2 反事实学习
parser.add_argument('--use_counterfactual', default=True, type=bool, help='是否使用反事实学习')
parser.add_argument('--intervened_type', default="random", type=str, help='反事实学习中干扰矩阵A方法, random表示随机')

#创新点4，动态梯度调整
parser.add_argument('--use_ogm', default=False, type=bool, help='是否使用动态梯度调整')

#loss函数的选择
parser.add_argument('--loss_fn',default="Triplet",type=str, help='选择loss函数, CrossEntropy表示交叉熵, Triplet表示三元组损失')

#训练细节数据
parser.add_argument('--run_num', default="1", type=str, help='实验的编号')
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--wd', default=1e-5,type=float)
parser.add_argument('--num_epochs', default=50,type=int,help='训练的epoch数,默认50, 训练将在15小时内完成')
parser.add_argument('--lr_steps', nargs='+', default=[20,30], type=int)
parser.add_argument('--gamma', default=0.1, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--dropout', default=0.3, type=float)
parser.add_argument('--batch_size', default=8, type=int)

#debug
parser.add_argument('--debug', default=1, type=int, help='是否开启debug模式, 0表示关闭, 1表示开启')
args = parser.parse_args()
if args.debug:
    wandb.init(config=args,
               project="PACS_modal_missing",
               name="exp1",
               mode="disabled"
               )
else:
    wandb.init(config=args,
                project="PACS_modal_missing",
                name=args.run_num,
                mode="offline"
                )
    
LOGGER_FILENAME = os.path.join(args.save_path, 'logs', args.run_num, 'log.txt')#记录训练过程的日志文件
CHECKPOINTS_SAVE_PATH = os.path.join(args.save_path, 'checkpoints', args.run_num)#记录checkpoints的地方
WANDB_PATH = os.path.join(args.save_path, 'wandb', args.run_num)#记录wandb的地方
VIS_PATH = os.path.join(args.save_path, 'vis', args.run_num)#记录vis的地方,主要是记录一些最终的可视化结果
def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
#确保以上四个文件夹存在
ensure_directory_exists(os.path.dirname(LOGGER_FILENAME))  # 对于log文件，我们需要创建其目录
ensure_directory_exists(CHECKPOINTS_SAVE_PATH)
ensure_directory_exists(WANDB_PATH)
ensure_directory_exists(VIS_PATH)
print("文件夹创建成功,保存路径均没问题")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = get_logger(LOGGER_FILENAME)#起一个loger

# Load the model,包括视频和音频
model = load_model("/home/lcs/T-PAMI2024_DCL/AudioCLIP/assets/ViT-B-16.pt", device, args = args)#加载ViT模型
logger.info("ViT Model loaded")
audio_model = AudioCLIPFinetune(pretrained=f'/home/lcs/T-PAMI2024_DCL/AudioCLIP/assets/{args.model_filename}')
for param in audio_model.parameters():#不需要微调
    param.requires_grad = False
model.audio_model = audio_model.audio#使用AudioCLIP模型的audio部分，但是不使用AudioCLIP的其他模态的编码

# 冻结视觉编码器和音频编码器的参数
name_list = []
pdb.set_trace()
for name in model.named_parameters():
    name_list.append(name[0])
for name, param in model.named_parameters():
    #TODO audio_model和visual以及transforer开头的参数分别代表什么意思？
    if name.startswith("audio_model") or name.startswith("visual") or name.startswith("transformer"):
        param.requires_grad = False #audiomodel冻结
    else:
        param.requires_grad = True#其他的全部有梯度，要训练的
        print(name)

model = model.to(device)

# Prepare the optimizer
#TODO 更加精细的优化策略
optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)#这俩参数有待调整
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.gamma)#有待调整

# Loss function
if args.loss_fn == "CrossEntropy":
    loss_fn = torch.nn.CrossEntropyLoss()
elif args.loss_fn == "Triplet":
    loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=nn.CosineSimilarity(), margin=1)
loss_fn = loss_fn.to(device)

##加载数据集的函数需要单独的拉进utils_extra.py中
# Load the Dataset
dataset_collecter, dataloader_collecter, dataset_size_collecter = load_dataset(args)

# Train the model
logger.info("Start training")
logger.info("Training size: {}, Validation size: {}, Test size: {}.".format(dataset_size_collecter["train"], dataset_size_collecter["val"], dataset_size_collecter["test"]))
best_acc = 0
for epoch in tqdm(range(args.num_epochs)):
    ## TODO 把训练过程抽象为一个函数，train/val/test通用
    # Train
    loss, acc = train_model('train', dataloader_collecter, model, optimizer, device, dataset_size_collecter, args, epoch, logger)  
    # Validation
    with torch.no_grad():
        loss, acc = train_model('val', dataloader_collecter, model, optimizer, device, dataset_size_collecter, args, epoch, logger)
    # Test
    with torch.no_grad():
        loss, acc = train_model('test', dataloader_collecter, model, optimizer, device, dataset_size_collecter, args, epoch, logger)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(CHECKPOINTS_SAVE_PATH, f"model_audioclip_{args.run_num}_best.pt"))
            logger.info("Best model saved to {}".format(CHECKPOINTS_SAVE_PATH))
    # Save the model,保存倒数后5个model和最好的那个model    
    if epoch >= args.num_epochs - 5:
        torch.save(model.state_dict(), os.path.join(CHECKPOINTS_SAVE_PATH, f"model_audioclip_{args.run_num}_{epoch}.pt"))
        logger.info("Model saved to {}".format(CHECKPOINTS_SAVE_PATH))
        