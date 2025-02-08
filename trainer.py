from utils_extra import compute_acc_loss
import wandb
def train_model(train_type, dataloader_collecter, model, optimizer, device, dataset_size_collecter, args, epoch, logger):
    '''
    input: train_type,dataloader_collecter, model, optimizer, device, data, dataset_size_collecter, args, epoch, logger, wandb
    output: loss, acc
    '''    
    total_correct = 0
    total_size = 0
    total_loss = 0
    
    
    iter_loss = 0 
    iter_total = 0
    iter_correct = 0
    iter_loss = 0 
    iter_total = 0
    iter_correct = 0
    iter_size = 0
    if train_type == "train":
        model.train()
    else:
        model.eval()
    ##TODO 我们希望根据不同的创新点能够输出不同的信息，所以需要开一个大的if
    if args.use_modal_share:
       pass 
    if args.use_vae:
        #需要保存的东西很多
        iter_record = dict()
        iter_record["o1_Recon_loss"] = 0
        iter_record["o1_kld_f"] = 0
        iter_record["o1_kld_z"] = 0
        iter_record["o2_Recon_loss"] = 0
        iter_record["o2_kld_f"] = 0
        iter_record["o2_kld_z"] = 0
        iter_record["o1_MI"] = 0
        iter_record["o2_MI"] = 0
        iter_record["loss_1_VAE"] = 0
        iter_record["loss_2_VAE"] = 0
    if args.use_counterfactual:
        pass
    if args.use_ogm:
        pass
    for i, data in enumerate(dataloader_collecter[train_type]):
        img1 = data["img1"].to(device)
        img2 = data["img2"].to(device)
        audio1 = data["audio1"].to(device)
        audio2 = data["audio2"].to(device)
        video1 = data["video1"].to(device)#视频实质上就是多张图片的List，20张
        video2 = data["video2"].to(device)
        token = data["tokens"].to(device)
        label = data["label"].to(device)
        #TODO 此处只接受Loss和传递出的特征，Loss和传递出的特征采用字典的形式展现
        ##直接输出预测的结果，然后由CPU统计正确率，Loss由GPU计算
        output, loss_dict, feature_dict = model(img1, audio1, img2, audio2, video1, video2, token, dataset_size_collecter[train_type],args, train_type)
        loss, correct = compute_acc_loss(output, label, args, loss_dict, feature_dict)
        ##梯度回传
        if train_type == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ##记录iter的正确率
        total_size += token.size(0)
        total_correct += correct
        total_loss += loss.item()

        iter_size += token.size(0)
        iter_correct += correct
        iter_loss += loss.item()
        if args.use_vae:
            iter_record["o1_Recon_loss"] += loss_dict['o1_Recon_loss']
            iter_record["o1_kld_f"] += loss_dict['o1_kld_f']
            iter_record["o1_kld_z"] += loss_dict['o1_kld_z']
            iter_record["o2_Recon_loss"] += loss_dict['o2_Recon_loss']
            iter_record["o2_kld_f"] += loss_dict['o2_kld_f']
            iter_record["o2_kld_z"] += loss_dict['o2_kld_z']
            iter_record["o1_MI"] += loss_dict['o1_MI']
            iter_record["o2_MI"] += loss_dict['o2_MI']
            iter_record["loss_1_VAE"] += loss_dict['loss_1_VAE']
            iter_record["loss_2_VAE"] += loss_dict['loss_2_VAE']
        if train_type == 'train':
            if i % 20 == 1:#每21个就打印一次
                logger.info("Epoch {}: Iter {}: Train loss: {}, Train acc: {}".format(epoch, i, iter_loss/iter_size*args.batch_size, iter_correct/iter_size))
                iter_loss = 0
                iter_size = 0
                iter_correct = 0
        else:
            pass
        #记录一些VAE等模块的Loss的数据，这里比较细，所以需要仔细的写一个函数来记录
    #直接记录Epoch的平均Loss
    wandb.log({train_type + "/loss": total_loss / dataset_size_collecter[train_type]})
    wandb.log({train_type + "/acc":  total_correct/dataset_size_collecter[train_type]})
    if args.use_vae:
        for key in iter_record.keys():
            wandb.log({train_type + "/" + key: iter_record[key] / dataset_size_collecter[train_type]})
    logger.info("Epoch {}: {}: loss: {}, acc: {}".format(epoch, train_type, total_loss / dataset_size_collecter[train_type], total_correct/dataset_size_collecter[train_type]))
    return total_loss / dataset_size_collecter[train_type], total_correct/dataset_size_collecter[train_type]