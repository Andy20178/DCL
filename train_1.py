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
from tqdm import tqdm 
import pdb

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)



'''
add train params
'''
parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', default="/data/lcs/PACS_data", type=str)#Need to be changed
parser.add_argument('--save_path', default="results", type=str)
parser.add_argument('--model_filename', default='AudioCLIP-Partial-Training.pt', type=str)

#Contribution in arxiv
parser.add_argument('--use_audio', default=True, type=bool)
parser.add_argument('--use_video',default=True,type=bool)
parser.add_argument('--use_image',default=False,type=bool)

parser.add_argument('--miss_modal', default="none", type=str, help='Select the type of missing modality.  \
                    "none" indicates no missing data, "audio" indicates missing audio, \
                    and "video" indicates missing video.')
parser.add_argument('--miss_ratio', default=0.5, type=float, help='Missing rate.')
parser.add_argument('--miss_noise', default="gaussian", type=str, help='Types of noise used to fill in the missing data: \
                                    gaussian: Gaussian noise, uniform: Uniform noise, random: Random noise, zero: Zero noise')
parser.add_argument('--use_modal_share', default=True, type=bool, help='Whether to use modal imputation')
parser.add_argument('--num_modal_class', default=2, type=int, help='class of modal')

#Contribution 1 in NeurIPS2023
parser.add_argument('--use_vae', default=True, type=bool, help='Whether to use VAE')
parser.add_argument('--vae_type', default="TransVAE", type=str)
parser.add_argument('--f_dim', default=256, type=int, help='Feature dimension')
parser.add_argument('--z_dim', default=32, type=int, help='Dimension of z')
parser.add_argument('--g_dim', default=128, type=int, help='Dimension of g, the dimension of each frame in VAE')
parser.add_argument('--rnn_size', default=256, type=int, help='Size of RNN')
parser.add_argument('--f_rnn_layers', default=1, type=int, help='Number of layers in f_rnn')
parser.add_argument('--num_frames', default=20, type=int, help='Number of frames extracted from the video')
parser.add_argument('--final_dim', default=512, type=int, help='Dimension of the final output')
parser.add_argument('--hidden_dim', default=256, type=int, help='Hidden dimension, all intermediate variables are of this dimension')

#Contribution 2 in NeurIPS2023
parser.add_argument('--use_knowledge', default=True, type=bool)
parser.add_argument('--use_static_knowledge', default=True, type=bool)
parser.add_argument('--use_dynamic_knowledge', default=True, type=bool)
parser.add_argument('--use_audio_knowledge', default=False, type=bool)
parser.add_argument('--use_image_knowledge', default=False, type=bool)
parser.add_argument('--use_video_knowledge', default=False, type=bool)
parser.add_argument('--top_k', default=5, type=int)
parser.add_argument('--use_counterfactual', default=True, type=bool)
parser.add_argument('--intervened_type', default="random", type=str)

parser.add_argument('--loss_fn',default="Triplet",type=str)

parser.add_argument('--run_num', default="1", type=str)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--wd', default=1e-5,type=float)
parser.add_argument('--num_epochs', default=50,type=int)
parser.add_argument('--lr_steps', nargs='+', default=[20,30], type=int)
parser.add_argument('--gamma', default=0.1, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--dropout', default=0.3, type=float)
parser.add_argument('--batch_size', default=8, type=int)

#debug
parser.add_argument('--debug', default=1, type=int)
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
    
LOGGER_FILENAME = os.path.join(args.save_path, 'logs', args.run_num, 'log.txt')  # Log file to record the training process
CHECKPOINTS_SAVE_PATH = os.path.join(args.save_path, 'checkpoints', args.run_num)  # Location to save checkpoints
WANDB_PATH = os.path.join(args.save_path, 'wandb', args.run_num)  # Location to save wandb logs
VIS_PATH = os.path.join(args.save_path, 'vis', args.run_num)  # Location to save visualizations, mainly to record some final visualization results

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Ensure the above four directories exist
ensure_directory_exists(os.path.dirname(LOGGER_FILENAME))  # For the log file, we need to create its directory
ensure_directory_exists(CHECKPOINTS_SAVE_PATH)
ensure_directory_exists(WANDB_PATH)
ensure_directory_exists(VIS_PATH)
print("Directories created successfully, all save paths are fine")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = get_logger(LOGGER_FILENAME)  # Start a logger

# Load the model, including video and audio
model = load_model("/home/lcs/T-PAMI2024_DCL/AudioCLIP/assets/ViT-B-16.pt", device, args=args)  # Load the ViT model
logger.info("ViT Model loaded")
audio_model = AudioCLIPFinetune(pretrained=f'/home/lcs/T-PAMI2024_DCL/AudioCLIP/assets/{args.model_filename}')
for param in audio_model.parameters():  # No need to fine-tune
    param.requires_grad = False
model.audio_model = audio_model.audio 

# Freeze the parameters of the visual encoder and audio encoder
name_list = []
pdb.set_trace()
for name in model.named_parameters():
    name_list.append(name[0])
for name, param in model.named_parameters():
    if name.startswith("audio_model") or name.startswith("visual") or name.startswith("transformer"):
        param.requires_grad = False  # Freeze audio model
    else:
        param.requires_grad = True  # All others have gradients and are to be trained
        print(name)

model = model.to(device)

# Prepare the optimizer
optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.gamma)

# Loss function
if args.loss_fn == "CrossEntropy":
    loss_fn = torch.nn.CrossEntropyLoss()
elif args.loss_fn == "Triplet":
    loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=nn.CosineSimilarity(), margin=1)
loss_fn = loss_fn.to(device)

# Load the Dataset
dataset_collecter, dataloader_collecter, dataset_size_collecter = load_dataset(args)

# Train the model
logger.info("Start training")
logger.info("Training size: {}, Validation size: {}, Test size: {}.".format(dataset_size_collecter["train"], dataset_size_collecter["val"], dataset_size_collecter["test"]))
best_acc = 0
for epoch in tqdm(range(args.num_epochs)):
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
    if epoch >= args.num_epochs - 5:
        torch.save(model.state_dict(), os.path.join(CHECKPOINTS_SAVE_PATH, f"model_audioclip_{args.run_num}_{epoch}.pt"))
        logger.info("Model saved to {}".format(CHECKPOINTS_SAVE_PATH))
        