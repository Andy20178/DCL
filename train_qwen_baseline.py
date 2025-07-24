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
from tqdm import tqdm   # Progress bar
import pdb
# from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor

'''
Set random seed
'''
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

'''
Add training arguments
'''
parser = argparse.ArgumentParser()
# Path settings
parser.add_argument('--data_dir', default="PACS_data", type=str)
parser.add_argument('--save_path', default="results", type=str,help='Used to save all results, including log, checkpoints, vis, and wandb')
parser.add_argument('--model_filename', default='AudioCLIP-Partial-Training.pt', type=str, help='Path to the pretrained AudioCLIP model')

# Data input parameters
parser.add_argument('--use_audio', action='store_true',default=False)
parser.add_argument('--use_video',default=True,type=bool)
parser.add_argument('--use_image',action='store_true',default=False)

# Model selection
# Innovation 1: Modal completion, modal_share
parser.add_argument('--miss_modal', default="None", type=str, help='Select the missing modality type, None means no missing, audio means missing audio, video means missing video')
parser.add_argument('--miss_ratio', default=0.5, type=float, help='Missing ratio')
parser.add_argument('--miss_noise', default="gaussian", type=str, help='Type of noise for missing modality, gaussian means Gaussian noise, uniform means uniform noise, random means random noise, zero means zero noise')
parser.add_argument('--use_modal_share', default=True, type=bool, help='Whether to use modal completion')
parser.add_argument('--num_modal_class', default=2, type=int, help='Number of modal completion classes')
# Innovation 2: Disentangled VAE
parser.add_argument('--use_vae', default=False, type=bool, help='Whether to use VAE')
parser.add_argument('--vae_type', default="TransVAE", type=str, help='Type of VAE, CDSVAE(NeurIPS2021), TransVAE(NeurIPS2023), S3VAE(CVPR2020), including different mutual information calculation methods')
parser.add_argument('--f_dim', default=256, type=int, help='Feature dimension')
parser.add_argument('--z_dim', default=32, type=int, help='Dimension of z')
parser.add_argument('--g_dim', default=128, type=int, help='Dimension of g, per-frame dimension in VAE')
parser.add_argument('--rnn_size', default=256, type=int, help='RNN size')
parser.add_argument('--f_rnn_layers', default=1, type=int, help='Number of f_rnn layers')
parser.add_argument('--num_frames', default=8, type=int, help='Number of frames sampled from video')
parser.add_argument('--final_dim', default=512, type=int, help='Final output dimension')
parser.add_argument('--hidden_dim', default=256, type=int, help='Hidden dimension, all intermediate variables use this dimension')

# Innovation 3: Counterfactual learning
## 3.1 Common knowledge extraction
parser.add_argument('--use_knowledge', default=False, type=bool, help='Whether to use common knowledge')
parser.add_argument('--use_static_knowledge', default=False, type=bool, help='Whether to use static common knowledge')
parser.add_argument('--use_dynamic_knowledge', default=False, type=bool, help='Whether to use dynamic common knowledge')
parser.add_argument('--use_audio_knowledge', default=False, type=bool, help='Whether to use audio common knowledge')
parser.add_argument('--use_image_knowledge', default=False, type=bool, help='Whether to use image common knowledge')
parser.add_argument('--use_video_knowledge', default=False, type=bool, help='Whether to use original video common knowledge')
parser.add_argument('--top_k', default=5, type=int, help='Top_k for common knowledge')
## 3.2 Counterfactual learning
parser.add_argument('--use_counterfactual', default=False, type=bool, help='Whether to use counterfactual learning')
parser.add_argument('--intervened_type', default="random", type=str, help='Intervention matrix A method in counterfactual learning, random means random')

# Innovation 4: Dynamic gradient adjustment
parser.add_argument('--use_ogm', default=False, type=bool, help='Whether to use dynamic gradient adjustment')

# Loss function selection
parser.add_argument('--loss_fn',default="Triplet",type=str, help='Select loss function, CrossEntropy for cross-entropy, Triplet for triplet loss')

# Training details
parser.add_argument('--run_num', default="exp_110", type=str, help='Experiment number')
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--wd', default=1e-5,type=float)
parser.add_argument('--num_epochs', default=50,type=int,help='Number of training epochs, default 50, training will finish within 15 hours')
parser.add_argument('--lr_steps', nargs='+', default=[20,30], type=int)
parser.add_argument('--gamma', default=0.1, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--dropout', default=0.3, type=float)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--Qwen2_5_Size', default='3B', type=str, help='Qwen2.5 model size, 3B, 7B, 32B')
parser.add_argument('--sim_type', default='cosine', type=str, help='Type of similarity calculation, cosine for cosine similarity, euclidean for Euclidean distance, manhattan for Manhattan distance')

# Debug
parser.add_argument('--debug', default=0, type=int, help='Whether to enable debug mode, 0 for off, 1 for on')
args = parser.parse_args()
# if args.debug:
#     wandb.init(config=args,
#                project="PACS_modal_missing",
#                name="exp1",
#                mode="disabled"
#                )
# else:
#     wandb.init(config=args,
#                 project="PACS_modal_missing",
#                 name=args.run_num,
#                 mode="offline"
#                 )
    
LOGGER_FILENAME = os.path.join(args.save_path, 'logs', args.run_num, 'log.txt') # Log file to record the training process
CHECKPOINTS_SAVE_PATH = os.path.join(args.save_path, 'checkpoints', args.run_num) # Directory to save checkpoints
WANDB_PATH = os.path.join(args.save_path, 'wandb', args.run_num) # Directory to save wandb files
VIS_PATH = os.path.join(args.save_path, 'vis', args.run_num) # Directory to save visualization results, mainly for final visualizations
def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
# Ensure the above four folders exist
ensure_directory_exists(os.path.dirname(LOGGER_FILENAME))  # For the log file, we need to create its directory
ensure_directory_exists(CHECKPOINTS_SAVE_PATH)
ensure_directory_exists(WANDB_PATH)
ensure_directory_exists(VIS_PATH)
print("Folders created successfully, all save paths are correct")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = get_logger(LOGGER_FILENAME) # Create a logger

model = load_model("assets/ViT-B-16.pt", device, args = args) # Load ViT model
logger.info("ViT Model loaded")
audio_model = AudioCLIPFinetune(pretrained=f'assets/{args.model_filename}')
for param in audio_model.parameters(): # No need to finetune
    param.requires_grad = False
model.audio_model = audio_model.audio # Use the audio part of the AudioCLIP model, but not the encoders for other modalities

# Freeze the parameters of the visual and audio encoders
name_list = []
# pdb.set_trace()
for name in model.named_parameters():
    name_list.append(name[0])
for name, param in model.named_parameters():
    # TODO: What do the parameters starting with audio_model, visual, and transformer mean?
    if name.startswith("audio_model") or name.startswith("visual") or name.startswith("transformer"):
        param.requires_grad = False # Freeze audiomodel
    else:
        param.requires_grad = True # All other parameters require gradients and will be trained
        print(name)

model = model.to(device)

# Prepare the optimizer
# TODO: More fine-grained optimization strategy
optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd) # These two parameters need to be adjusted
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.gamma) # Needs to be adjusted

# Loss function
if args.loss_fn == "CrossEntropy":
    loss_fn = torch.nn.CrossEntropyLoss()
elif args.loss_fn == "Triplet":
    loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=nn.CosineSimilarity(), margin=1)
loss_fn = loss_fn.to(device)

# The function to load the dataset needs to be moved into utils_extra.py
# Load the Dataset
# import pdb; pdb.set_trace()
dataset_collecter, dataloader_collecter, dataset_size_collecter = load_dataset(args)

# Train the model
logger.info("Start training")
logger.info("Training size: {}, Validation size: {}, Test size: {}.".format(dataset_size_collecter["train"], dataset_size_collecter["val"], dataset_size_collecter["test"]))
best_acc = 0
for epoch in tqdm(range(args.num_epochs)):
    # TODO: Abstract the training process into a function, common for train/val/test
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
    # Save the model, save the last 5 models and the best model    
    if epoch >= args.num_epochs - 5:
        torch.save(model.state_dict(), os.path.join(CHECKPOINTS_SAVE_PATH, f"model_audioclip_{args.run_num}_{epoch}.pt"))
        logger.info("Model saved to {}".format(CHECKPOINTS_SAVE_PATH))