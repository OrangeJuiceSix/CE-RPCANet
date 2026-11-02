# train_dp.py
import os
import os.path as osp
import time
import datetime
from argparse import ArgumentParser

import numpy as np
import torch
import torch.utils.data as Data

try:
    from tensorboardX import SummaryWriter
except Exception:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore

import torch.nn as nn

from models import get_model
from utils.data import *
from utils.loss import SoftLoULoss
from utils.lr_scheduler import *
from utils.metrics import SegmentationMetricTPFNFP
from utils.my_pd_fa import my_PD_FA
from utils.pd_fa import PD_FA
from utils.logger import setup_logger

# 允许 TF32（几乎“白嫖”提速）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")  # PyTorch 2.x 推荐


def set_seeds(seed: int):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = ArgumentParser(description="Implement of RPCANets (DataParallel)")

    # Dataset
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--dataset", type=str, default="nudt")

    # Training
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warm-up-epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gpu", type=str, default="0,1")  # e.g. "0" or "0,1"
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr-scheduler", type=str, default="poly")

    # Net
    parser.add_argument("--net-name", type=str, default="ce_rpcanet")

    # Save / log
    parser.add_argument("--save-iter-step", type=int, default=100)
    parser.add_argument("--log-per-iter", type=int, default=10)
    parser.add_argument("--base-dir", type=str, default="./result/")

    args = parser.parse_args()

    if args.gpu is not None and args.gpu != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # e.g. "0,1"

    args.time_name = time.strftime("%Y%m%dT%H-%M-%S", time.localtime(time.time()))
    args.folder_name = f"{args.time_name}_{args.net_name}_{args.dataset}"
    args.save_folder = osp.join(args.base_dir, args.folder_name)
    os.makedirs(args.save_folder, exist_ok=True)

    if args.seed != 0:
        set_seeds(args.seed)

    args.logger = setup_logger("Robust PCA Networks", args.save_folder, 0, filename="log.txt")
    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.iter_num = 0



        # ======================
        # dataset
        # ======================
        if args.dataset == "sirstaug":
            trainset = SirstAugDataset(base_dir=r"./datasets/sirst_aug", mode="train", base_size=args.base_size)
            valset = SirstAugDataset(base_dir=r"./datasets/sirst_aug", mode="test", base_size=args.base_size)
        elif args.dataset == "irstd1k":
            trainset = IRSTD1kDataset(base_dir=r"./datasets/IRSTD-1k", mode="train", base_size=args.base_size)
            valset = IRSTD1kDataset(base_dir=r"./datasets/IRSTD-1k", mode="test", base_size=args.base_size)
        elif args.dataset == "nudt":
            trainset = NUDTDataset(base_dir=r"./datasets/NUDT-SIRST", mode="train", base_size=args.base_size)
            valset = NUDTDataset(base_dir=r"./datasets/NUDT-SIRST", mode="test", base_size=args.base_size)
        elif args.dataset == "sirst":
            trainset = SirstDataset(base_dir=r"./datasets/sirst", mode="train", base_size=args.base_size)
            valset = SirstDataset(base_dir=r"./datasets/sirst", mode="val", base_size=args.base_size)
        elif args.dataset == "drive":
            trainset = DriveDatasetTrain(base_dir=r"./datasets/DRIVE", mode="train",
                                         base_size=args.base_size, patch_size=args.crop_size)
            valset = DriveDatasetTest(base_dir=r"./datasets/DRIVE", mode="test", base_size=args.base_size)
        elif args.dataset == "CHASEDB1":
            trainset = CHASEDB1DatasetTrain(base_dir=r"./datasets/CHASEDB1", mode="train",
                                            base_size=args.base_size, patch_size=args.crop_size)
            valset = CHASEDB1DatasetTest(base_dir=r"./datasets/CHASEDB1", mode="test", base_size=args.base_size)
        elif args.dataset == "stare":
            trainset = STAREDatasetTrain(base_dir=r"./datasets/STARE", mode="train",
                                         base_size=args.base_size, patch_size=args.crop_size)
            valset = STAREDatasetTest(base_dir=r"./datasets/STARE", mode="test", base_size=args.base_size)
        elif args.dataset == "densesirst":
            trainset = DenseSirstDataset(base_dir=r"./datasets/DenseSIRST", mode="train", base_size=args.base_size)
            valset = DenseSirstDataset(base_dir=r"./datasets/DenseSIRST", mode="val", base_size=args.base_size)
        else:
            raise NotImplementedError

        # DataLoader 优化
        self.train_data_loader = Data.DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,              
            pin_memory=True,
            persistent_workers=True,    
            prefetch_factor=4,          
            drop_last=True              
        )

        self.val_data_loader = Data.DataLoader(
            valset,
            batch_size=args.batch_size,
            shuffle=False,              
            num_workers=8,              
            pin_memory=True,
            persistent_workers=True,   
            prefetch_factor=4,
            drop_last=False             
        )


        # self.train_data_loader = Data.DataLoader(
        #     trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
        # )
        # self.val_data_loader = Data.DataLoader(
        #     valset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
        # )

        self.iter_per_epoch = len(self.train_data_loader)
        self.max_iter = args.epochs * self.iter_per_epoch

        # ======================
        # logger
        # ======================
        self.logger = args.logger
        self.logger.info(args)

        # ======================
        # device + multi-gpu
        # ======================
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        visible_cnt = torch.cuda.device_count() if self.use_cuda else 0
        self.logger.info(f"Using device: {self.device}, cuda_available={self.use_cuda}, visible_gpu_num={visible_cnt}")

        # ======================
        # model
        # ======================
        net = get_model(args.net_name).to(self.device)
        if self.use_cuda and visible_cnt > 1:
            net = nn.DataParallel(net, device_ids=list(range(visible_cnt)))
            self.logger.info(f"Using nn.DataParallel with {visible_cnt} GPUs (visible).")
        self.net = net

        # ======================
        # loss
        # ======================
        self.softiou = SoftLoULoss() 
        self.mse = torch.nn.MSELoss()

        # BCE 用 logits
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0], device=self.device))

        # ======================
        # lr scheduler
        # ======================
        self.scheduler = LR_Scheduler_Head(args.lr_scheduler, args.lr, args.epochs, len(self.train_data_loader), lr_step=10)

        # ======================
        # optimizer
        # ======================
        small_lr_params, normal_params = [], []
        for n, p in self.net.named_parameters():
            if ("epsilon" in n) or ("alpha" in n):
                small_lr_params.append(p)
            else:
                normal_params.append(p)

        self.optimizer = torch.optim.Adam(
            [
                {"params": normal_params, "lr": args.lr},
                {"params": small_lr_params, "lr": args.lr * 0.1},
            ],
            lr=args.lr,
        )

        # ======================
        # metrics
        # ======================
        self.metric = SegmentationMetricTPFNFP(nclass=1)
        self.best_iou = 0.0
        self.best_fmeasure = 0.0

        # ======================
        # tb
        # ======================
        self.writer = SummaryWriter(log_dir=args.save_folder)
        self.writer.add_text(args.folder_name, "Args:%s, " % args)

    @staticmethod
    def logits_to_prob_for_eval(logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.clamp(logits, -50, 50))

    def _get_state_dict(self):
        if isinstance(self.net, nn.DataParallel):
            return self.net.module.state_dict()
        return self.net.state_dict()

    def training(self):
        start_time = time.time()
        base_log = (
            "Epoch-Iter: [{:d}/{:d}]-[{:d}/{:d}] || Lr: {:.6f} || "
            "Loss: {:.4f}={:.4f}+{:.4f}+{:.4f} || Cost Time: {} || ETA: {}"
        )

        for epoch in range(self.args.epochs):
            for i, (data, labels) in enumerate(self.train_data_loader):
                self.net.train()
                self.scheduler(self.optimizer, i, epoch, self.best_iou)

                data = data.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).float() 

                out_D, out_T = self.net(data)
                loss_softiou = self.softiou(out_T, labels)   # logits
                loss_bce = self.bce(out_T, labels)           # logits
                loss_mse = self.mse(out_D, data)
                loss_all = loss_softiou + 0.2 * loss_bce + 0.01 * loss_mse

                self.optimizer.zero_grad(set_to_none=True)
                loss_all.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.optimizer.step()

                self.iter_num += 1

                cost_string = str(datetime.timedelta(seconds=int(time.time() - start_time)))
                eta_seconds = ((time.time() - start_time) / max(self.iter_num, 1)) * (self.max_iter - self.iter_num)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                self.writer.add_scalar("Train/Loss_All", loss_all.item(), self.iter_num)
                self.writer.add_scalar("Train/Loss_SoftIoU", loss_softiou.item(), self.iter_num)
                self.writer.add_scalar("Train/Loss_BCE", loss_bce.item(), self.iter_num)
                self.writer.add_scalar("Train/Loss_MSE", loss_mse.item(), self.iter_num)
                self.writer.add_scalar("Train/LR", self.optimizer.param_groups[0]["lr"], self.iter_num)

                if self.iter_num % self.args.log_per_iter == 0:
                    self.logger.info(
                        base_log.format(
                            epoch + 1,
                            self.args.epochs,
                            (i + 1),
                            self.iter_per_epoch,
                            self.optimizer.param_groups[0]["lr"],
                            loss_all.item(),
                            loss_softiou.item(),
                            loss_bce.item(),
                            loss_mse.item(),
                            cost_string,
                            eta_string,
                        )
                    )

                if (self.iter_num % self.args.save_iter_step) == 0 or (i + 1) == self.iter_per_epoch:
                    self.validation()

    def validation(self):
        self.metric.reset()
        self.net.eval()

        eval_my = my_PD_FA()
        eval_pd = PD_FA()

        base_log = (
            "Data: {:s}, IoU: {:.4f}/{:.4f}, F1: {:.4f}/{:.4f}, "
            "PD_FA: (PD={:.4f}, FA={:.6f}), my_PD_FA: (PD={:.4f}, FA={:.6f})"
        )

        for _, (data, labels) in enumerate(self.val_data_loader):
            data = data.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).float()

            with torch.no_grad():
                _, out_T = self.net(data)  # logits

            # ========= 1) IoU/F1：严格复刻原口径 =========
            # 你原来的 batch_tp_fp_fn: predict = (output > 0)
            out_T_cpu = out_T.detach().cpu()
            labels_cpu = labels.detach().cpu()
            self.metric.update(labels_cpu, out_T_cpu) 

            # ========= 2) PD/FA：用概率 =========
            pred_prob = self.logits_to_prob_for_eval(out_T).detach().cpu()  # [0,1]

            B = pred_prob.shape[0]
            for b in range(B):
                pred_np = pred_prob[b].squeeze(0).numpy()
                lab_np = labels_cpu[b].squeeze(0).numpy()
                H, W = lab_np.shape[-2], lab_np.shape[-1]

                eval_pd.update(pred_np, lab_np, size=(H, W), thr=0.5, match_thresh=3.0)
                if np.max(lab_np) > 0:
                    eval_my.update(pred_np, lab_np)

        iou, prec, recall, fmeasure = self.metric.get()
        pd_PD, pd_FA = eval_pd.get()
        myPD, myFA = eval_my.get()

        # save
        state_dict = self._get_state_dict()
        torch.save(state_dict, osp.join(self.args.save_folder, "latest.pkl"))
        if iou > self.best_iou:
            self.best_iou = iou
            torch.save(state_dict, osp.join(self.args.save_folder, "best.pkl"))
        if fmeasure > self.best_fmeasure:
            self.best_fmeasure = fmeasure

        # tensorboard
        self.writer.add_scalar("Val/IoU", iou, self.iter_num)
        self.writer.add_scalar("Val/F1", fmeasure, self.iter_num)
        self.writer.add_scalar("Val/PD_FA/PD", pd_PD, self.iter_num)
        self.writer.add_scalar("Val/PD_FA/FA", pd_FA, self.iter_num)
        self.writer.add_scalar("Val/my_PD_FA/PD", myPD, self.iter_num)
        self.writer.add_scalar("Val/my_PD_FA/FA", myFA, self.iter_num)
        self.writer.add_scalar("Best/IoU", self.best_iou, self.iter_num)
        self.writer.add_scalar("Best/F1", self.best_fmeasure, self.iter_num)

        self.logger.info(
            base_log.format(self.args.dataset, iou, self.best_iou, fmeasure, self.best_fmeasure, pd_PD, pd_FA, myPD, myFA)
        )


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    trainer.training()
    print("Best mIoU: %.5f, Best Fmeasure: %.5f\n\n" % (trainer.best_iou, trainer.best_fmeasure))
