import os
import yaml
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import sys
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))
from tools.datasets import WSIWithCluster, mixup, fusion_features_weighted, sample_feats
from tools.general import AverageMeter, CSVWriter, EarlyStop, increment_path, BestVariable, init_seeds, save_checkpoint, load_json
from models import ppo, clam
from tools.losses import NT_Xent


def create_save_dir(args):
    dir1 = args.dataset
    dir2 = f'train_1'
    dir3 = f'CLAM_SB'
    dir4 = f'stage_{args.train_stage}'
    args.save_dir = str(Path(args.base_save_dir) / dir1 / dir2 / dir3 / dir4)
    print(f"save_dir: {args.save_dir}")

def get_datasets(args):
    indices = load_json(args.data_split_json)[args.fold_id]['train']
    train_set = WSIWithCluster(
        data_csv=args.data_csv+'/fold_'+args.fold_id+'.csv',
        indices=indices,
        shuffle=True,
    )
    print('args.num_clusters', args.num_clusters)
    return train_set, train_set.patch_dim, len(train_set)


def create_model(args, dim_patch):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.device == 'cpu' else 'cpu')
    model = clam.CLAM_SB(
        gate=True,
        size_arg=args.size_arg,
        dropout=True,
        k_sample=args.k_sample,
        n_classes=args.projection_dim,
        subtyping=True,
        in_dim=dim_patch
    )
    model = ppo.CL(model, projection_dim=args.projection_dim, n_features=model.classifiers.in_features)
    fc = ppo.Full_layer(args.feature_num, args.fc_hidden_dim, args.fc_rnn, args.projection_dim)#projection_dim=128
    ppo = None

    # Load checkpoint
    if args.train_stage == 1:
        pass
    elif args.train_stage == 2:
        checkpoint = torch.load(str(Path(args.save_dir).parent / 'stage_1' / 'model_best.pth.tar'))
        model.load_state_dict(checkpoint['model_state_dict'])
        fc.load_state_dict(checkpoint['fc'])
        state_dim = args.model_dim
        ppo = ppo.PPO(dim_patch, state_dim, args.policy_hidden_dim, args.policy_conv,
                        action_std=args.action_std,
                        lr=args.ppo_lr,
                        gamma=args.ppo_gamma,
                        K_epochs=args.K_epochs,
                        action_size=args.num_clusters)
    elif args.train_stage == 3:
        checkpoint = torch.load(str(Path(args.save_dir).parent / 'stage_2' / 'model_best.pth.tar'))
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
    model = torch.nn.DataParallel(model).to(device)
    fc = fc.to(device)
    assert model is not None, "creating model failed. "
    print(f"model Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"fc Total params: {sum(p.numel() for p in fc.parameters()) / 1e6:.2f}M")
    return model, fc, ppo

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


def train(args, train_set, model, fc, ppo, criterion, optimizer, scheduler, tb_writer, save_dir):
    save_dir = Path(save_dir)
    best_train_loss = BestVariable(order='min')
    header = ['epoch', 'train', 'best_epoch', 'best_train']
    losses_csv = CSVWriter(filename=save_dir / 'losses.csv', header=header)
    results_csv = CSVWriter(filename=save_dir / 'results.csv', header=['epoch', 'final_epoch', 'final_loss'])
    early_stop = EarlyStop(max_num_accordance=args.patience) if args.patience is not None else None

    if args.train_stage == 2:
        model.eval()
        fc.eval()
    else:
        model.train()
        fc.train()
    memory_list = [ppo.Memory(), ppo.Memory()]
    for epoch in range(args.epochs):
        print(f"Stage: {args.train_stage}, lr:")
        if optimizer is not None:
            for k, group in enumerate(optimizer.param_groups):
                print(f"group[{k}]: {group['lr']}")
        train_set.shuffle()
        length = len(train_set)
        losses = [AverageMeter() for _ in range(args.T)]
        reward_list = [AverageMeter() for _ in range(args.T - 1)]
        # print("num_data:", args.num_data)
        progress_bar = tqdm(range(args.num_data))
        feat_list, cluster_list, step = [], [], 0
        batch_idx = 0

        for data_idx in progress_bar:
            loss_list = []
            feat, cluster, *_ = train_set[data_idx % length]
            assert len(feat.shape) == 2, f"{feat.shape}"
            feat = feat.unsqueeze(0).to(args.device)
            feat_list.append(feat)
            cluster_list.append(cluster)
            step += 1
            if step == args.batch_size:
                F_Ck = sample_feats(feat_list, cluster_list, feat_size=args.feat_size)
                action_sequences = [torch.ones((len(feat_list), args.num_clusters), device=feat_list[0].device) / args.num_clusters for _ in
                                    range(2)]
                x_views = [fusion_features_weighted(F_Ck, a, feat_size=args.feat_size) for a in
                           action_sequences]
                x_views = [mixup(x, args.alpha)[0] for x in x_views]
                if args.train_stage != 2:
                    outputs, states = model(x_views)
                    outputs = [fc(o, restart=True) for o in outputs]
                else:
                    with torch.no_grad():
                        outputs, states = model(x_views)
                        outputs = [fc(o, restart=True) for o in outputs]

                loss = criterion(outputs[0], outputs[1])
                loss_list.append(loss)
                losses[0].update(loss.data.item(), len(feat_list))
                similarity_last = torch.cosine_similarity(outputs[0], outputs[1]).view(1, -1)
                for patch_step in range(1, args.T):
                    if args.train_stage == 1:
                        action_sequences = [torch.ones((len(feat_list), args.num_clusters),
                                                       device=feat_list[0].device) / args.num_clusters for _ in range(2)]
                    else:
                        if patch_step == 1:
                            action_sequences = [ppo.select_action(s.to(0), m, restart_batch=True) for s, m in
                                                zip(states, memory_list)]
                        else:
                            action_sequences = [ppo.select_action(s.to(0), m) for s, m in zip(states, memory_list)]

                    x_views = [fusion_features_weighted(F_Ck, a, feat_size=args.feat_size) for a in action_sequences]
                    x_views = [mixup(x, args.alpha)[0] for x in x_views]
                    if args.train_stage != 2:
                        outputs, states = model(x_views)
                        outputs = [fc(o, restart=False) for o in outputs]
                    else:
                        with torch.no_grad():
                            outputs, states = model(x_views)
                            outputs = [fc(o, restart=False) for o in outputs]

                    loss = criterion(outputs[0], outputs[1])
                    loss_list.append(loss)
                    losses[patch_step].update(loss.data.item(), len(feat_list))
                    similarity = torch.cosine_similarity(outputs[0], outputs[1]).view(1, -1)
                    reward = similarity_last - similarity
                    similarity_last = similarity
                    reward_list[patch_step - 1].update(reward.data.mean(), len(feat_list))
                    for m in memory_list:
                        m.rewards.append(reward)

                loss = sum(loss_list) / args.T
                if args.train_stage != 2:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    for m in memory_list:
                        ppo.update(m)

                for m in memory_list:
                    m.clear_memory()

                feat_list, cluster_list, step = [], [], 0
                batch_idx += 1

                progress_bar.set_description(
                    f"Train Epoch: {epoch + 1:2}/{args.epochs:2}. Iter: {batch_idx:3}/{args.eval_step:3}. "
                    f"Loss: {losses[-1].avg:.4f}. "
                )
                progress_bar.update()
        progress_bar.close()

        if scheduler is not None and epoch >= args.warmup:
            scheduler.step()
        train_loss = losses[-1].avg
        if tb_writer is not None:
            tb_writer.add_scalar('train/1.train_loss', train_loss, epoch)

        is_best = best_train_loss.compare(train_loss, epoch + 1, inplace=True)
        state = {
            'epoch': epoch + 1,
            'model_state_dict': model.module.state_dict(),
            'fc': fc.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer else None,
            'ppo_optimizer': ppo.optimizer.state_dict() if ppo else None,
            'policy': ppo.policy.state_dict() if ppo else None,
        }
        save_checkpoint(state, is_best, str(save_dir))
        losses_csv.write_row([epoch + 1, train_loss, best_train_loss.epoch, best_train_loss.best])
        results_csv.write_row([epoch + 1, best_train_loss.epoch, best_train_loss.best])
        print(f"Loss: {train_loss:.4f}, Best: {best_train_loss.best:.4f}, Epoch: {best_train_loss.epoch:2}\n")

        if early_stop is not None:
            early_stop.update(best_train_loss.best)
            if early_stop.is_stop():
                break

    if tb_writer is not None:
        tb_writer.close()


def run(args):
    init_seeds(args.seed)
    if args.save_dir is None:
        create_save_dir(args)
    else:
        args.save_dir = str(Path(args.base_save_dir) / args.save_dir)
    args.save_dir = increment_path(Path(args.save_dir), exist_ok=True, sep='_')  # increment run
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    if not args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        args.device = torch.device('cuda' if torch.cuda.is_available() and not args.device == 'cpu' else 'cpu')

    else:
        args.device = torch.device('cpu')

    # Dataset
    train_set, dim_patch, train_length = get_datasets(args)
    args.num_data = train_length * args.data_repeat
    args.eval_step = int(args.num_data / args.batch_size)
    print(f"train_length: {train_length}, epoch_step: {args.num_data}, eval_step: {args.eval_step}")

    model, fc, ppo = create_model(args, dim_patch)
    criterion = NT_Xent(args.batch_size, args.temperature)
    optimizer = get_optimizer(args, model, fc)
    scheduler = get_scheduler(args, optimizer)

    with open(Path(args.save_dir) / 'args.yaml', 'w') as fp:
        yaml.dump(vars(args), fp, sort_keys=False)
    print(args, '\n')
    tb_writer = SummaryWriter(args.save_dir) if args.use_tensorboard else None
    train(args, train_set, model, fc, ppo, criterion, optimizer, scheduler, tb_writer, args.save_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='AD',
                        help="dataset name")
    parser.add_argument('--fold_id', type=str, default='0',
                        help="fold id")
    parser.add_argument('--data_csv', type=str,help="the .csv filepath used")
    parser.add_argument('--data_split_json', type=str)
    parser.add_argument('--device', default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--seed', type=int, default=2021,help="random state")
    parser.add_argument('--num_clusters', type=int, default=6)
    parser.add_argument('--base_save_dir', type=str, default='./fold_results_AD')
    parser.add_argument('--data_repeat', type=int, default=10,
                        help="contrastive learning need more iteration to train, the arg is to repeat data training for one epoch")
    parser.add_argument('--feat_size', default=1024, type=int)
    parser.add_argument('--train_stage', default=1, type=int,
                        help="select training stage \
                              stage-1 : warm-up \
                              stage-2 : learn to select patches with RL \
                              stage-3 : finetune")
    parser.add_argument('--T', default=6, type=int,
                        help="maximum length of the sequence of RNNs")
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'],
                        help="specify the optimizer used, default Adam")
    parser.add_argument('--scheduler', type=str, default=None, choices=[None, 'StepLR', 'CosineAnnealingLR'],
                        help="specify the lr scheduler used, default None")
    parser.add_argument('--batch_size', type=int, default=128,
                        help="the batch size for training")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--backbone_lr', default=1e-4, type=float,
                        help='the learning rate for MIL encoder')
    parser.add_argument('--fc_lr', default=1e-4, type=float,
                        help='the learning rate for FC')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help="the temperature coefficient of contrastive loss")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help="the momentum of SGD optimizer")
    parser.add_argument('--nesterov', action='store_true', default=True)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--warmup', default=0, type=float,
                        help="the number of epoch for training without lr scheduler, if scheduler is not None")
    parser.add_argument('--wdecay', default=1e-5, type=float,
                        help="the weight decay of optimizer")
    parser.add_argument('--patience', type=int, default=None,
                        help="if the loss not change during `patience` epochs, the training will early stop")
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--projection_dim', type=int, default=128)
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
    parser.add_argument('--size_arg', type=str, default='small', choices=['small', 'big'])
    parser.add_argument('--k_sample', type=int, default=8)
    parser.add_argument('--use_tensorboard', action='store_true', default=False)
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()
    run(args)



if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    torch.set_num_threads(1)
    main()
