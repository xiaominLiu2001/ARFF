import os
import copy
import yaml
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
# 添加项目根目录到 sys.path
import sys
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))
from tools.datasets import WSIWithCluster, fusion_features_weighted, sample_feats
from tools.general import AverageMeter, CSVWriter, EarlyStop, increment_path, BestVariable, accuracy, init_seeds, \
    load_json, get_metrics, get_score, save_checkpoint, save_equal_checkpoint,save_valid_checkpoint
from models import ppo, clam

import time


def create_save_dir(args):
    dir1 = args.dataset
    dir2 = f'train_2_cls'
    dir3 = f'CLAM_SB'
    dir4 = f'stage_{args.train_stage}'
    args.save_dir = str(Path(args.base_save_dir) / dir1 / dir2 / dir3 / dir4)
    print(f"save_dir: {args.save_dir}")

def get_datasets(args):
    indices = load_json(args.data_split_json)[args.fold_id]
    train_set = WSIWithCluster(
        data_csv=args.data_csv+'/fold_'+args.fold_id+'.csv',
        indices=indices['train'],
        num_sample_patches=args.feat_size,
        shuffle=True)
    valid_set = WSIWithCluster(
        data_csv=args.data_csv+'/fold_'+args.fold_id+'.csv',
        indices=indices['val'],
        num_sample_patches=args.feat_size,
        shuffle=False)
    test_set = WSIWithCluster(
        data_csv=args.data_csv+'/fold_'+args.fold_id+'.csv',
        indices=indices['val'],
        num_sample_patches=args.feat_size,
        shuffle=False)
    # args.num_clusters = train_set.num_clusters
    return {'train': train_set, 'valid': valid_set, 'test': test_set}, train_set.patch_dim, len(train_set)


def create_model(args, dim_patch):
    model = clam.CLAM_SB(
        gate=True,
        size_arg=args.size_arg,
        dropout=True,
        k_sample=args.k_sample,
        n_classes=args.num_classes,
        subtyping=True,
        in_dim=dim_patch
    )
    args.feature_num = dim_patch
    fc = ppo.Full_layer(512, args.fc_hidden_dim, args.fc_rnn, args.num_classes)
    ppo = None

    if args.train_stage == 1:
        assert args.checkpoint_pretrained is not None and Path(
            args.checkpoint_pretrained).exists(), f"{args.checkpoint_pretrained} is not exists!"

        checkpoint = torch.load(args.checkpoint_pretrained)
        model_state_dict = checkpoint['model_state_dict']
        for k in list(model_state_dict.keys()):
            if k.startswith('encoder') and not k.startswith('encoder.fc') and not k.startswith('encoder.classifiers'):
                model_state_dict[k[len('encoder.'):]] = model_state_dict[k]
            del model_state_dict[k]
        msg_model = model.load_state_dict(model_state_dict, strict=False)
        print(f"msg_model missing_keys: {msg_model.missing_keys}")

    elif args.train_stage == 2:
        checkpoint_stage = str(Path(args.save_dir).parent / 'stage_1' / 'model_best.pth.tar')
        checkpoint = torch.load(checkpoint_stage)
        model.load_state_dict(checkpoint['model_state_dict'])
        fc.load_state_dict(checkpoint['fc'])

        assert args.checkpoint_pretrained is not None and Path(
            args.checkpoint_pretrained).exists(), f"{args.checkpoint_pretrained} is not exists!"
        checkpoint = torch.load(args.checkpoint_pretrained)
        state_dim = args.model_dim
        ppo = ppo.PPO(dim_patch, state_dim, args.policy_hidden_dim, args.policy_conv,
                        action_std=args.action_std,
                        lr=args.ppo_lr,
                        gamma=args.ppo_gamma,
                        K_epochs=args.K_epochs,
                        action_size=args.num_clusters)
        ppo.policy.load_state_dict(checkpoint['policy'])
        ppo.policy_old.load_state_dict(checkpoint['policy'])

    elif args.train_stage == 3:
        checkpoint_stage = str(Path(args.save_dir).parent / 'stage_2' / 'model_best.pth.tar')
        checkpoint = torch.load(checkpoint_stage)
        model.load_state_dict(checkpoint['model_state_dict'])
        fc.load_state_dict(checkpoint['fc'])

        state_dim = args.model_dim
        ppo = ppo.PPO(dim_patch, state_dim, args.policy_hidden_dim, args.policy_conv,
                        action_std=args.action_std,
                        lr=args.ppo_lr,
                        gamma=args.ppo_gamma,
                        K_epochs=args.K_epochs,
                        action_size=args.num_clusters)
        ppo.policy.load_state_dict(checkpoint['policy'])
        ppo.policy_old.load_state_dict(checkpoint['policy'])
    else:
        raise ValueError

    model = torch.nn.DataParallel(model).cuda()
    fc = fc.cuda()
    assert model is not None, "creating model failed. "
    print(f"model Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"fc Total params: {sum(p.numel() for p in fc.parameters()) / 1e6:.2f}M")
    return model, fc, ppo


def get_criterion(args):
    if args.loss == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError(f"args.loss error, error value is {args.loss}.")
    return criterion


def get_optimizer(args, model, fc):
    if args.train_stage != 2:
        params = [{'params': model.parameters(), 'lr': args.backbone_lr},
                  {'params': fc.parameters(), 'lr': args.fc_lr}]
        if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(params,
                                        lr=0,  # specify in params
                                        momentum=args.momentum,
                                        nesterov=args.nesterov,
                                        weight_decay=args.wdecay)
        elif args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(params, betas=(args.beta1, args.beta2), weight_decay=args.wdecay)
        else:
            raise NotImplementedError
    else:
        optimizer = None
        args.epochs = 30
    return optimizer


def get_scheduler(args, optimizer):
    if optimizer is None:
        return None
    if args.scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    elif args.scheduler == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup, eta_min=1e-6)
    elif args.scheduler is None:
        scheduler = None
    else:
        raise ValueError
    return scheduler


def train_CLAM(args, epoch, train_set, model, fc, ppo, memory, criterion, optimizer, scheduler):
    length = len(train_set)
    train_set.shuffle()
    losses = [AverageMeter() for _ in range(args.T)]
    top1 = [AverageMeter() for _ in range(args.T)]
    reward_list = [AverageMeter() for _ in range(args.T - 1)]

    if args.train_stage == 2:
        model.eval()
        fc.eval()
    else:
        model.train()
        fc.train()

    progress_bar = tqdm(range(args.num_data))
    feat_list, cluster_list, label_list, step = [], [], [], 0
    batch_idx = 0
    labels_list, outputs_list = [], []
    for data_idx in progress_bar:
        loss_list = []
        feat, cluster, label, case_id = train_set[data_idx % length]
        assert len(feat.shape) == 2, f"{feat.shape}"
        feat = feat.unsqueeze(0).to(args.device)
        label = label.unsqueeze(0).to(args.device)
        feat_list.append(feat)
        cluster_list.append(cluster)
        label_list.append(label)
        step += 1
        if step == args.batch_size or data_idx == args.num_data - 1:
            labels = torch.cat(label_list)
            F_Ck = sample_feats(feat_list, cluster_list, feat_size=args.feat_size)
            action_sequence = torch.ones((len(feat_list), args.num_clusters), device=feat_list[0].device) / args.num_clusters
            feats = fusion_features_weighted(F_Ck, action_sequence, feat_size=args.feat_size)
            if args.train_stage != 2:
                outputs, states, result_dict = model(feats, label=labels, instance_eval=True)
                outputs = fc(outputs, restart=True)
            else:
                with torch.no_grad():
                    outputs, states, result_dict = model(feats, label=labels, instance_eval=True)
                    outputs = fc(outputs, restart=True)

            loss = args.bag_weight * criterion(outputs, labels) + (1 - args.bag_weight) * result_dict['instance_loss']
            loss_list.append(loss)
            losses[0].update(loss.data.item(), len(feat_list))
            acc = accuracy(outputs, labels, topk=(1,))[0]
            top1[0].update(acc.item(), len(feat_list))

            confidence_last = torch.gather(F.softmax(outputs.detach(), 1), dim=1, index=labels.view(-1, 1)).view(1, -1)
            for patch_step in range(1, args.T):
                if args.train_stage == 1:
                    action_sequence = torch.ones((len(feat_list), args.num_clusters),
                                                 device=feat_list[0].device) / args.num_clusters
                else:
                    if patch_step == 1:
                        action_sequence = ppo.select_action(states.to(0), memory, restart_batch=True)
                    else:
                        action_sequence = ppo.select_action(states.to(0), memory)

                feats = fusion_features_weighted(F_Ck, action_sequence, feat_size=args.feat_size)
                if args.train_stage != 2:
                    outputs, states, result_dict = model(feats, label=labels, instance_eval=True)
                    outputs = fc(outputs, restart=False)
                else:
                    with torch.no_grad():
                        outputs, states, result_dict = model(feats, label=labels, instance_eval=True)
                        outputs = fc(outputs, restart=False)
                loss = args.bag_weight * criterion(outputs, labels) + (1 - args.bag_weight) * result_dict[
                    'instance_loss']
                loss_list.append(loss)
                losses[patch_step].update(loss.data.item(), len(feat_list))
                acc = accuracy(outputs, labels, topk=(1,))[0]
                top1[patch_step].update(acc.item(), len(feat_list))
                confidence = torch.gather(F.softmax(outputs.detach(), 1), dim=1, index=labels.view(-1, 1)).view(1, -1)
                reward = confidence - confidence_last
                confidence_last = confidence
                reward_list[patch_step - 1].update(reward.data.mean(), len(feat_list))
                memory.rewards.append(reward)

            loss = sum(loss_list) / args.T
            if args.train_stage != 2:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                ppo.update(memory)
            memory.clear_memory()
            torch.cuda.empty_cache()

            labels_list.append(labels.detach())
            outputs_list.append(outputs.detach())
            feat_list, cluster_list, label_list, step = [], [], [], 0
            batch_idx += 1
            progress_bar.set_description(
                f"Train Epoch: {epoch + 1:2}/{args.epochs:2}. Iter: {batch_idx:3}/{args.eval_step:3}. "
                f"Loss: {losses[-1].avg:.4f}. Acc: {top1[-1].avg:.4f}"
            )
            progress_bar.update()

    progress_bar.close()
    if scheduler is not None and epoch >= args.warmup:
        scheduler.step()
    labels = torch.cat(labels_list)
    outputs = torch.cat(outputs_list)
    acc, auc, precision, recall, f1_score = get_metrics(outputs, labels)

    return losses[-1].avg, acc, auc, precision, recall, f1_score


def test_CLAM(args, test_set, model, fc, ppo, memory, criterion):
    losses = [AverageMeter() for _ in range(args.T)]
    reward_list = [AverageMeter() for _ in range(args.T - 1)]
    model.eval()
    fc.eval()
    with torch.no_grad():
        feat_list, cluster_list, label_list, case_id_list, step = [], [], [], [], 0
        start_time = time.time()
        for data_idx, (feat, cluster, label, case_id) in enumerate(test_set):
            feat = feat.unsqueeze(0).to(args.device)
            label = label.unsqueeze(0).to(args.device)
            feat_list.append(feat)
            cluster_list.append(cluster)
            label_list.append(label)
            case_id_list.append(case_id)

        loss_list = []
        labels = torch.cat(label_list)
        F_Ck = sample_feats(feat_list, cluster_list, feat_size=args.feat_size)
        action_sequence = torch.ones((len(feat_list), args.num_clusters),
                                     device=feat_list[0].device) / args.num_clusters
        feats = fusion_features_weighted(F_Ck, action_sequence, feat_size=args.feat_size)
        outputs, states, result_dict = model(feats, label=labels, instance_eval=True)
        outputs = fc(outputs, restart=True)
        ins_loss = 0
        for r in result_dict:
            ins_loss = ins_loss + r['instance_loss']
        ins_loss = ins_loss / len(feat_list)
        loss = args.bag_weight * criterion(outputs, labels) + (1 - args.bag_weight) * ins_loss
        loss_list.append(loss)
        confidence_last = torch.gather(F.softmax(outputs.detach(), 1), dim=1, index=labels.view(-1, 1)).view(1, -1)
        for patch_step in range(1, args.T):
            if args.train_stage == 1:
                action = torch.ones((len(feat_list), args.num_clusters),
                                             device=feat_list[0].device) / args.num_clusters
            else:
                if patch_step == 1:
                    action = ppo.select_action(states.to(0), memory, restart_batch=True)
                else:
                    action = ppo.select_action(states.to(0), memory)

            feats = fusion_features_weighted(F_Ck, action, feat_size=args.feat_size)
            outputs, states, result_dict = model(feats, label=labels, instance_eval=True)
            outputs = fc(outputs, restart=False)
            ins_loss = 0
            for r in result_dict:
                ins_loss = ins_loss + r['instance_loss']
            ins_loss = ins_loss / len(feat_list)
            loss = args.bag_weight * criterion(outputs, labels) + (1 - args.bag_weight) * ins_loss
            loss_list.append(loss)
            losses[patch_step].update(loss.data.item(), len(feat_list))

            confidence = torch.gather(F.softmax(outputs.detach(), 1), dim=1, index=labels.view(-1, 1)).view(1, -1)
            reward = confidence - confidence_last
            confidence_last = confidence

            reward_list[patch_step - 1].update(reward.data.mean(), len(feat_list))
            memory.rewards.append(reward)
        end_time = time.time()  # 记录推理结束时间
        inference_time = end_time - start_time
        AVG_INF_TIME = inference_time / len(case_id_list)
        print(f"Average Inference Time: {AVG_INF_TIME:.4f} seconds")
        memory.clear_memory()
        acc, auc, precision, recall, f1_score = get_metrics(outputs, labels)
    return losses[-1].avg, acc, auc, precision, recall, f1_score, outputs, labels, case_id_list

def train(args, train_set, valid_set, test_set, model, fc, ppo, memory, criterion, optimizer, scheduler, tb_writer):
    save_dir = args.save_dir
    best_train_acc = BestVariable(order='max')
    best_valid_acc = BestVariable(order='max')
    best_test_acc = BestVariable(order='max')
    best_train_auc = BestVariable(order='max')
    best_valid_auc = BestVariable(order='max')
    best_test_auc = BestVariable(order='max')
    best_train_loss = BestVariable(order='min')
    best_valid_loss = BestVariable(order='min')
    best_test_loss = BestVariable(order='min')
    best_score = BestVariable(order='max')
    tmp_best_val_acc = BestVariable(order='max')
    tmp_best_test_acc = BestVariable(order='max')
    final_loss, final_acc, final_auc, final_precision, final_recall, final_f1_score, final_epoch = 0., 0., 0., 0., 0., 0., 0
    header = ['epoch', 'train', 'valid', 'test', 'best_train', 'best_valid', 'best_test']
    losses_csv = CSVWriter(filename=Path(save_dir) / 'losses.csv', header=header)
    accs_csv = CSVWriter(filename=Path(save_dir) / 'accs.csv', header=header)
    aucs_csv = CSVWriter(filename=Path(save_dir) / 'aucs.csv', header=header)
    results_csv = CSVWriter(filename=Path(save_dir) / 'results.csv',
                            header=['epoch', 'final_epoch', 'final_loss', 'final_acc', 'final_auc', 'final_precision',
                                    'final_recall', 'final_f1_score'])

    best_model = copy.deepcopy({'state_dict': model.state_dict()})
    early_stop = None

    for epoch in range(args.epochs):
        print(f"Training Stage: {args.train_stage}, lr:")
        if optimizer is not None:
            for k, group in enumerate(optimizer.param_groups):
                print(f"group[{k}]: {group['lr']}")

        train_loss, train_acc, train_auc, train_precision, train_recall, train_f1_score = \
            train_CLAM(args, epoch, train_set, model, fc, ppo, memory, criterion, optimizer, scheduler)
        valid_loss, valid_acc, valid_auc, valid_precision, valid_recall, valid_f1_score, *_ = \
            test_CLAM(args, valid_set, model, fc, ppo, memory, criterion)
        test_loss, test_acc, test_auc, test_precision, test_recall, test_f1_score, *_ = \
            test_CLAM(args, test_set, model, fc, ppo, memory, criterion)

        if tb_writer is not None:
            tb_writer.add_scalar('train/1.train_loss', train_loss, epoch)
            tb_writer.add_scalar('test/2.test_loss', valid_loss, epoch)
        is_best = False
        is_equal = False
        is_best_valid=False #
        if tmp_best_val_acc.isequal(valid_acc) == 1 :
            if tmp_best_test_acc.isequal(test_acc) == 2:
                is_best_valid = True
            else:
                is_best = True
                tmp_best_val_acc.isequal(valid_acc,True)
                tmp_best_test_acc.isequal(test_acc,True)

        elif tmp_best_val_acc.isequal(valid_acc)== 0 :
            print("valid_acc--isequal--")
            if tmp_best_test_acc.isequal(test_acc) == 1:
                is_best = True
                tmp_best_val_acc.isequal(valid_acc, True)
                tmp_best_test_acc.isequal(test_acc, True)
            elif tmp_best_test_acc.isequal(test_acc) == 0:
                is_equal = True

        if is_best:
            final_epoch = epoch + 1
            final_loss = test_loss
            final_acc = test_acc
            final_auc = test_auc

        best_train_acc.compare(train_acc, epoch + 1, inplace=True)
        best_valid_acc.compare(valid_acc, epoch + 1, inplace=True)
        best_test_acc.compare(test_acc, epoch + 1, inplace=True)
        best_train_loss.compare(train_loss, epoch + 1, inplace=True)
        best_valid_loss.compare(valid_loss, epoch + 1, inplace=True)
        best_test_loss.compare(test_loss, epoch + 1, inplace=True)
        best_train_auc.compare(train_auc, epoch + 1, inplace=True)
        best_valid_auc.compare(valid_auc, epoch + 1, inplace=True)
        best_test_auc.compare(test_auc, epoch + 1, inplace=True)
        state = {
            'epoch': epoch + 1,
            'model_state_dict': model.module.state_dict(),
            'fc': fc.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer else None,
            'ppo_optimizer': ppo.optimizer.state_dict() if ppo else None,
            'policy': ppo.policy.state_dict() if ppo else None,
        }
        if is_best:
            best_model = copy.deepcopy(state)
            save_checkpoint(state, is_best, str(save_dir))
        if is_equal:
            best_model = copy.deepcopy(state)
            save_equal_checkpoint(state, str(save_dir),epoch,test_acc)
        if is_best_valid:
            best_model = copy.deepcopy(state)
            save_valid_checkpoint(state, str(save_dir),epoch,valid_acc)

        losses_csv.write_row([epoch + 1, train_loss, valid_loss, test_loss,
                              (best_train_loss.best, best_train_loss.epoch),
                              (best_valid_loss.best, best_valid_loss.epoch),
                              (best_test_loss.best, best_test_loss.epoch)])
        accs_csv.write_row([epoch + 1, train_acc, valid_acc, test_acc,
                            (best_train_acc.best, best_train_acc.epoch),
                            (best_valid_acc.best, best_valid_acc.epoch),
                            (best_test_acc.best, best_test_acc.epoch)])
        aucs_csv.write_row([epoch + 1, train_auc, valid_auc, test_auc,
                            (best_train_auc.best, best_train_auc.epoch),
                            (best_valid_auc.best, best_valid_auc.epoch),
                            (best_test_auc.best, best_test_auc.epoch)])
        results_csv.write_row(
            [epoch + 1, final_epoch, test_loss, test_acc, test_auc, test_precision, test_recall, test_f1_score])

        print(
            f"Train acc: {train_acc:.4f}, Best: {best_train_acc.best:.4f}, Epoch: {best_train_acc.epoch:2}, "
            f"AUC: {train_auc:.4f}, Best: {best_train_auc.best:.4f}, Epoch: {best_train_auc.epoch:2}, "
            f"Loss: {train_loss:.4f}, Best: {best_train_loss.best:.4f}, Epoch: {best_train_loss.epoch:2}\n"
            f"Valid acc: {valid_acc:.4f}, Best: {best_valid_acc.best:.4f}, Epoch: {best_valid_acc.epoch:2}, "
            f"AUC: {valid_auc:.4f}, Best: {best_valid_auc.best:.4f}, Epoch: {best_valid_auc.epoch:2}, "
            f"Loss: {valid_loss:.4f}, Best: {best_valid_loss.best:.4f}, Epoch: {best_valid_loss.epoch:2}\n"
            f"Test  acc: {test_acc:.4f}, Best: {best_test_acc.best:.4f}, Epoch: {best_test_acc.epoch:2}, "
            f"AUC: {test_auc:.4f}, Best: {best_test_auc.best:.4f}, Epoch: {best_test_auc.epoch:2}, "
            f"Loss: {test_loss:.4f}, Best: {best_test_loss.best:.4f}, Epoch: {best_test_loss.epoch:2}\n"
            f"Final Epoch: {final_epoch:2}, Final acc: {final_acc:.4f}, Final AUC: {final_auc:.4f}, Final Loss: {final_loss:.4f}\n"
        )

        if early_stop is not None:
            early_stop.update((best_valid_loss.best, best_valid_acc.best, best_valid_auc.best))
            if early_stop.is_stop():
                break

    if tb_writer is not None:
        tb_writer.close()

    return best_model


def test(args, test_set, model, fc, ppo, memory, criterion):
    model.eval()
    fc.eval()
    with torch.no_grad():
        loss, acc, auc, precision, recall, f1_score, outputs_tensor, labels_tensor, case_id_list = \
            test_CLAM(args, test_set, model, fc, ppo, memory, criterion)
        prob = torch.softmax(outputs_tensor, dim=1)
        _, pred = torch.max(prob, dim=1)
        preds = pd.DataFrame(columns=['label', 'pred', 'correct', *[f'prob{i}' for i in range(prob.shape[1])]])
        for i in range(len(case_id_list)):
            preds.loc[case_id_list[i]] = [
                labels_tensor[i].item(),
                pred[i].item(),
                labels_tensor[i].item() == pred[i].item(),
                *[prob[i][j].item() for j in range(prob.shape[1])],
            ]
        preds.index.rename('case_id', inplace=True)

    return loss, acc, auc, precision, recall, f1_score, preds


def run(args):
    init_seeds(args.seed)

    if args.save_dir is None:
        create_save_dir(args)
    else:
        args.save_dir = str(Path(args.base_save_dir) / args.save_dir)
    args.save_dir = increment_path(Path(args.save_dir), exist_ok=True, sep='_')
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    if not args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device('cpu')

    datasets, dim_patch, train_length = get_datasets(args)
    print("dim_patch", dim_patch)
    args.num_data = train_length
    args.eval_step = int(args.num_data / args.batch_size)
    print(f"train_length: {train_length}, epoch_step: {args.num_data}, eval_step: {args.eval_step}")

    model, fc, ppo = create_model(args, dim_patch)
    criterion = get_criterion(args)
    optimizer = get_optimizer(args, model, fc)
    scheduler = get_scheduler(args, optimizer)

    # Save arguments
    with open(Path(args.save_dir) / 'args.yaml', 'w') as fp:
        yaml.dump(vars(args), fp, sort_keys=False)
    print(args, '\n')
    tb_writer = SummaryWriter(args.save_dir) if args.use_tensorboard else None

    memory = ppo.Memory()
    best_model = train(args, datasets['train'], datasets['valid'], datasets['test'], model, fc, ppo, memory, criterion,
                       optimizer, scheduler, tb_writer)
    model.module.load_state_dict(best_model['model_state_dict'])
    fc.load_state_dict(best_model['fc'])
    if ppo is not None:
        ppo.policy.load_state_dict(best_model['policy'])
    loss, acc, auc, precision, recall, f1_score, preds = \
        test(args, datasets['test'], model, fc, ppo, memory, criterion)

    preds.to_csv(str(Path(args.save_dir) / 'pred.csv'))
    final_res = pd.DataFrame(columns=['loss', 'acc', 'auc', 'precision', 'recall', 'f1_score'])
    final_res.loc[f'seed{args.seed}'] = [loss, acc, auc, precision, recall, f1_score]
    final_res.to_csv(str(Path(args.save_dir) / 'final_res.csv'))
    print(f'{final_res}\nPredicted Ending.\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='AD',
                        help="dataset name")
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--fold_id', type=str, default='0',
                        help="fold id")
    parser.add_argument('--data_csv', type=str, help="the .csv filepath used")
    parser.add_argument('--data_split_json', type=str)
    parser.add_argument('--feat_size', default=1024, type=int)
    parser.add_argument('--train_stage', default=1, type=int)
    parser.add_argument('--checkpoint_pretrained', type=str, default=None,
                        help='path to the pretrained checkpoint (for finetune and linear)')
    parser.add_argument('--T', default=6, type=int,
                        help="maximum length of the sequence of RNNs")
    parser.add_argument('--loss', default='CrossEntropyLoss', type=str, choices=LOSSES,
                        help='loss name')
    parser.add_argument('--scheduler', type=str, default='CosineAnnealingLR', choices=[None, 'StepLR', 'CosineAnnealingLR'],
                        help="specify the lr scheduler used, default None")
    parser.add_argument('--picked_method', type=str, default='acc',
                        help="the metric of pick best model from validation dataset")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="the batch size for training")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--backbone_lr', default=5e-5, type=float)
    parser.add_argument('--fc_lr', default=2e-5, type=float)
    parser.add_argument('--num_clusters', type=int, default=6)
    parser.add_argument('--device', default='0',help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--seed', type=int, default=2021,help="random state")
    parser.add_argument('--base_save_dir', type=str, default='./results_AD')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'],
                        help="specify the optimizer used, default Adam")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', action='store_true', default=True)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--warmup', default=0, type=float,
                        help="the number of epoch for training without lr scheduler, if scheduler is not None")
    parser.add_argument('--wdecay', default=1e-5, type=float,
                        help="the weight decay of optimizer")
    parser.add_argument('--patience', type=int, default=None,
                        help="if the loss not change during `patience` epochs, the training will early stop")
    parser.add_argument('--model_dim', type=int, default=512)
    parser.add_argument('--policy_hidden_dim', type=int, default=512)
    parser.add_argument('--policy_conv', action='store_true', default=False)
    parser.add_argument('--action_std', type=float, default=0.5)
    parser.add_argument('--ppo_lr', type=float, default=0.00001)
    parser.add_argument('--ppo_gamma', type=float, default=0.1)
    parser.add_argument('--K_epochs', type=int, default=3)
    parser.add_argument('--feature_num', type=int, default=512)
    parser.add_argument('--fc_hidden_dim', type=int, default=1024)
    parser.add_argument('--fc_rnn', action='store_true', default=True)
    parser.add_argument('--load_fc', action='store_true', default=False)
    parser.add_argument('--size_arg', type=str, default='small', choices=['small', 'big'])
    parser.add_argument('--k_sample', type=int, default=8)
    parser.add_argument('--bag_weight', type=float, default=0.7)
    parser.add_argument('--use_tensorboard', action='store_true', default=False)
    parser.add_argument('--save_dir', type=str, default=None,
                        help="specify the save directory to save experiment results, default None."
                             "If not specify, the directory will be create by function create_save_dir(args)")
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    torch.set_num_threads(1)
    MODELS = ['CLAM_SB']
    LOSSES = ['CrossEntropyLoss']
    TRAIN = {'CLAM_SB': train_CLAMtrain_CLAM,}
    TEST = {'CLAM_SB': test_CLAM,}
    main()
