import datetime
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import *
from utils.augment import Cutout
from utils.data_loader import get_statistics
from utils.onlinesampler import OnlineSampler, OnlineTestSampler
from utils.train_utils import select_model, select_optimizer, select_scheduler

logger = logging.getLogger()
mp.set_sharing_strategy('file_system')


class _Trainer():
    def __init__(self, *args, **kwargs) -> None:

        # 将所有命令行 / 配置参数保存到 trainer 实例上，后续各个方法都能通过 self.xxx 直接访问。
        self.kwargs = kwargs
        self.__dict__.update(kwargs)

        # 运行时记录信息，用于日志和周期性评估。
        self.start_time = time.time()
        self.eval_period = np.inf if self.eval_period < 0 else self.eval_period

        # 为部分方法启用“按样本数推进”的内部 step 调度（不依赖真实任务边界）。
        method_name = getattr(self, "method", None)
        step_aware_methods = {"dualprompt", "mvp", "flyprompt"}
        if method_name in step_aware_methods:
            # step_num 必须大于 1；如果没有提供或 <=0，则默认使用 n_tasks。
            self.step_num = getattr(self, "step_num", None)
            if self.step_num is None or self.step_num <= 0:
                if hasattr(self, "n_tasks"):
                    self.step_num = self.n_tasks
            if self.step_num is not None and self.step_num <= 1:
                raise ValueError(f"step_num must be > 1, got {self.step_num}")
        else:
            # 其他方法继续使用原来的“按任务编号”调度方式。
            self.step_num = None

        # 这些变量会在知道数据集大小后再完整初始化。
        # current_step：当前内部 expert / stage 的编号。
        # current_step_seen_samples：当前 step 已经消耗了多少样本。
        # samples_per_step：在 _init_internal_step_scheduler 中根据总样本数计算得到。
        self.current_step = 0
        self.current_step_seen_samples = 0
        self.samples_per_step = None

        # 分布式训练相关设置。
        # world_size 表示所有 GPU 上的总 worker 数。
        # 如果是分布式训练，会相应缩小每个进程的 batch size。
        self.world_size = 1
        self.ngpus_per_nodes = torch.cuda.device_count()
        if "WORLD_SIZE" in os.environ and os.environ["WORLD_SIZE"] != '':
            self.world_size  = int(os.environ["WORLD_SIZE"]) * self.ngpus_per_nodes
        else:
            self.world_size  = self.world_size * self.ngpus_per_nodes

        self.distributed = self.world_size > 1
        self.dist_backend = 'nccl'
        self.dist_url = 'env://'
        if self.distributed:
            self.batchsize = self.batchsize // self.world_size

        self.log_dir = f"{self.log_path}/logs/{self.dataset}/{self.note}"

        # 确保训练开始前输出目录已经存在。
        os.makedirs(self.log_dir, exist_ok=True)
        self.progress_log_path = os.path.join(self.log_dir, f"training_progress_seed_{getattr(self, 'rnd_seed', 'na')}.jsonl")
        self.final_checkpoint_path = os.path.join(self.log_dir, f"final_model_seed_{getattr(self, 'rnd_seed', 'na')}.pth")
        self.current_task_id = 0

        return

    def setup_distributed_dataset(self):

        # DATASETS 是 datasets/__init__.py 里的“名字 -> 数据集类”注册表。
        self.datasets = DATASETS

        # 数据集统计量会影响归一化和类别数等关键设置。
        # 对 ViT 来说，这里统一把输入尺寸覆盖成 224，不管原始数据集是什么。
        mean, std, n_classes, inp_size, in_channels = get_statistics(dataset=self.dataset)
        inp_size = 224 # 为 ViT 统一覆盖输入尺寸
        self.n_classes = n_classes
        self.inp_size = inp_size
        self.mean = mean
        self.std = std

        train_transform = []
        self.cutmix = "cutmix" in self.transforms
        if "cutout" in self.transforms:
            train_transform.append(Cutout(size=16))
        if "autoaug" in self.transforms:
            if 'cifar' in self.dataset:
                train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('cifar10')))
            elif 'imagenet' in self.dataset:
                train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('imagenet')))
            elif 'wikiart' in self.dataset:
                train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('imagenet')))
            elif 'svhn' in self.dataset:
                train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('svhn')))

        # 组合训练阶段的数据增强流程，最终输出是可以直接送入 ViT 的归一化张量。
        self.train_transform = transforms.Compose([
                lambda x: (x * 255).to(torch.uint8),
                transforms.Resize((inp_size, inp_size)),
                transforms.RandomCrop(inp_size, padding=4),
                transforms.RandomHorizontalFlip(),
                *train_transform,
                lambda x: x.float() / 255,
                # transforms.ToTensor() 已经不需要，因为前面先转成了 tensor。
                transforms.Normalize(mean, std),])
        logger.info(f"Using train-transforms {train_transform}")
        # test_transform 用于数据集返回 PIL 图像的情况。
        self.test_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),])

        # 为“输入本身已经是 tensor”的情况准备一个兼容版测试 transform。
        # 某些方法会直接把内存中的 tensor 传给评估函数，而不是 PIL 图像。
        self.test_transform_tensor = transforms.Compose([
                transforms.Resize((inp_size, inp_size)),
                # 这里不再做 ToTensor()，因为输入已经是 tensor 了。
                transforms.Normalize(mean, std),])

        if 'imagenet' in self.dataset or 'cub' in self.dataset or 'car' in self.dataset:
            self.load_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size)),
                transforms.ToTensor()])
        else:
            self.load_transform = transforms.ToTensor()

        # 构建训练 / 测试数据集。自定义数据集可能会忽略 download=True，
        # 但 torchvision 的数据集通常可以自动下载。
        self.train_dataset = self.datasets[self.dataset](root=self.data_dir, train=True,  download=True, transform=self.load_transform)
        self.online_iter_dataset = OnlineIterDataset(self.train_dataset, 1)
        self.test_dataset = self.datasets[self.dataset](root=self.data_dir, train=False, download=True, transform=self.test_transform)

        # OnlineSampler 会按照 n / m 规则为每个任务生成样本下标流。
        # train_dataloader 会从 set_task() 选中的任务下标里取数据。
        _r = dist.get_rank() if self.distributed else None       # 非分布式时这里就是 None
        _w = dist.get_world_size() if self.distributed else None # 非分布式时这里就是 None
        self.train_sampler = OnlineSampler(self.online_iter_dataset, self.n_tasks, self.m, self.n, self.rnd_seed, 0, self.rnd_NM, _w, _r)
        self.train_dataloader = DataLoader(self.online_iter_dataset, batch_size=self.batchsize, sampler=self.train_sampler, pin_memory=False, num_workers=0)
        self.test_sampler = OnlineTestSampler(self.test_dataset, [], _w, _r)

        # 持续学习状态：
        # exposed_classes：到目前为止已经见过的全局类别 id。
        # mask：[n_classes] 维的 logits 加性掩码，没见过的类别保持为 -inf。
        self.seen = 0
        self.exposed_classes = []
        self.disjoint_classes = self.train_sampler.disjoint_classes
        self.mask = torch.zeros(self.n_classes, device=self.device) - torch.inf

    def setup_distributed_model(self):

        logger.info(f"Building model: {self.method}")
        # select_model 会根据 method / backbone 解析出正确的模型类。
        self.model = select_model(self.method, self.backbone, self.n_classes, self.n_tasks, self.kwargs).to(self.device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # model_without_ddp 用来统一访问真实模型主体，单卡和 DDP 都能用同一种写法。
        self.model.to(self.device)
        self.model_without_ddp = self.model

        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)
            self.model._set_static_graph()
            self.model_without_ddp = self.model.module
        # 损失函数可以被具体方法模型里的 loss_fn 覆盖。
        self.criterion = getattr(self.model_without_ddp, "loss_fn", nn.CrossEntropyLoss(reduction="mean"))
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.lr_gamma = 0.99995 if 'imagenet' in self.dataset else 0.9999
        self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)

        n_params = sum(p.numel() for p in self.model_without_ddp.parameters())
        logger.info(f"Total Parameters :\t{n_params}")
        n_params = sum(p.numel() for p in self.model_without_ddp.parameters() if p.requires_grad)
        learnables = [n for n, p in self.model_without_ddp.named_parameters() if p.requires_grad]
        logger.info(f"Learnable Parameters :\t{n_params}")
        logger.info(learnables)
        logger.info("")

    def run(self):
        # main.py 创建 trainer 之后会调用这里。
        # profile 模式只跑一次短迭代；普通模式则跑完整训练流程。
        if self.profile:
            self.profile_worker(0)
        else:
            # 分布式启动
            if self.ngpus_per_nodes > 1:
                mp.spawn(self.main_worker, nprocs=self.ngpus_per_nodes, join=True)
            else:
                self.main_worker(0)

    def _init_internal_step_scheduler(self):
        """根据训练集大小初始化内部 step 调度。

        使用场景：FlyPrompt / DualPrompt / MVP 这类方法，
        不想完全依赖 benchmark 的任务边界，而是希望“看了多少样本”
        就切换到下一个内部 expert / stage。

        例子：如果训练集有 50000 张图，step_num=5，
        那么这里会估计每个内部 step 约处理 10000 张图，
        训练时每累计到这个数量，就进入下一个 step。
        """
        if getattr(self, "step_num", None) is None:
            return
        if self.step_num <= 1:
            # 这在 __init__ 里已经校验过，这里再防御一下。
            raise ValueError(f"step_num must be > 1, got {self.step_num}")
        if not hasattr(self, "total_samples"):
            return
        if self.total_samples <= 0:
            return

        # 用训练集总大小来估算每个内部 step 大约包含多少样本。
        # 这是近似值，因为 sampler 可能会重新排序样本。
        # 例如：total_samples=50000，step_num=5，那么 samples_per_step=10000。
        self.samples_per_step = max(1, self.total_samples // self.step_num)
        self.current_step = 0
        self.current_step_seen_samples = 0

    def _maybe_advance_internal_step(self, batch_size: int):
        """只根据已经看到的样本数量推进内部 step 计数器。

        使用场景：当一个 batch 训练完后，训练器会调用这个函数，
        用于判断“是不是已经看够一段样本，可以切到下一个 expert 了”。
        这对在线场景很重要，因为你可能并不知道真实任务边界。

        例子：batch_size=64，samples_per_step=1000，
        那么大约看了 16 个 batch 后，就会触发一次 step 前进。
        如果模型实现了 process_task_count()，这里还会顺带通知模型更新专家状态。
        """
        if getattr(self, "step_num", None) is None:
            return
        if getattr(self, "samples_per_step", None) is None:
            return
        if self.step_num <= 1 or batch_size <= 0:
            return

        self.current_step_seen_samples += batch_size
        while self.current_step < self.step_num - 1 and self.current_step_seen_samples >= self.samples_per_step:
            self.current_step_seen_samples -= self.samples_per_step
            self.current_step += 1

            # 通知模型已经跨过一个内部 step 边界。
            # 使用 expert bank 的方法通常会在这里做响应（例如初始化新 expert）。
            model_obj = getattr(self, "model_without_ddp", None)
            if model_obj is None:
                model_obj = getattr(self, "model", None)
            if model_obj is not None and hasattr(model_obj, "process_task_count"):
                model_obj.process_task_count()


    def main_worker(self, gpu) -> None:
        # 单 GPU 对应一个 worker 进程；多 GPU 时，mp.spawn() 会为每张卡启动一个进程。
        # 使用场景：这是“真正开始训练”的地方，
        # 入口 main.py 只是创建 trainer，这里才会把数据、模型、循环串起来。
        # ========= 分布式训练初始化 =========
        self.gpu    = gpu % self.ngpus_per_nodes
        self.device = torch.device(self.gpu)
        if self.distributed:
            self.local_rank = self.gpu
            if 'SLURM_PROCID' in os.environ.keys():
                self.rank = int(os.environ['SLURM_PROCID']) * self.ngpus_per_nodes + self.gpu
                logger.info(f"| Init Process group {os.environ['SLURM_PROCID']} : {self.local_rank}")
            else :
                self.rank = self.gpu
                logger.info(f"| Init Process group 0 : {self.local_rank}")
            if 'MASTER_ADDR' not in os.environ.keys():
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '12702'
            torch.cuda.set_device(self.gpu)
            time.sleep(self.rank * 0.1) # 防止端口冲突
            dist.init_process_group(backend=self.dist_backend, init_method=self.dist_url,
                                    world_size=self.world_size, rank=self.rank)
            torch.distributed.barrier()
            self.setup_for_distributed(self.is_main_process())
        else:
            pass

        # 为每个 worker 设置随机种子，保证可复现性。
        if self.rnd_seed is not None:
            random.seed(self.rnd_seed)
            np.random.seed(self.rnd_seed)
            torch.manual_seed(self.rnd_seed)
            torch.cuda.manual_seed(self.rnd_seed)
            torch.cuda.manual_seed_all(self.rnd_seed) # 多 GPU 时也要同步设置
            cudnn.deterministic = True
            logger.info(
                'You have chosen to seed training. '
                'This will turn on the CUDNN deterministic setting, '
                'which can slow down your training considerably! '
                'You may see unexpected behavior when restarting '
                'from checkpoints.'
            )
        cudnn.benchmark = False

        self.setup_distributed_dataset()
        self.total_samples = len(self.train_dataset)
        self._init_internal_step_scheduler()

        logger.info(f"[1] Select a GCL method ({self.method})")
        self.setup_distributed_model()

        # =========== 增量训练主循环 ==========
        # task_records：每个任务结束时的最终指标。
        # eval_results：训练过程中周期性评估得到的在线指标。
        logger.info(f"[2] Incrementally training {self.n_tasks} tasks")
        task_records = defaultdict(list)
        eval_results = defaultdict(list)
        samples_cnt = 0

        num_eval = self.eval_period
        num_report = 2000
        report_period = 500

        for task_id in range(self.n_tasks):
            self.current_task_id = task_id

            logger.info("\n")
            logger.info("#" * 50)
            logger.info(f"# Task {task_id} iteration")
            logger.info("#" * 50 + "\n")
            logger.info("[2-1] Prepare a datalist for the current task")

            # 切换 sampler 到当前 benchmark 任务，并让方法做任务相关准备。
            self.train_sampler.set_task(task_id)
            self.online_before_task(task_id)
            for epoch in range(self.num_epochs):
                logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
                for i, (images, labels, idx) in enumerate(self.train_dataloader):
                    # samples_cnt 表示全局已经消费了多少训练样本，
                    # 用来控制日志打印和评估频率。
                    samples_cnt += images.size(0) * self.world_size

                    # 核心的“每个 batch 怎么训练”逻辑，由子类实现。
                    loss, acc = self.online_step(images, labels, idx)

                    if samples_cnt + images.size(0) * self.world_size > num_report:
                        self.report_training(samples_cnt, loss, acc)
                        num_report += report_period

                    if samples_cnt + images.size(0) * self.world_size > num_eval:
                        with torch.no_grad():
                            # 在线设置下，只在当前已见类别上评估。
                            test_sampler = OnlineTestSampler(self.test_dataset, self.exposed_classes)
                            test_dataloader = DataLoader(self.test_dataset, batch_size=self.batchsize*2, sampler=test_sampler, num_workers=self.n_worker)
                            eval_dict = self.online_evaluate(test_dataloader)
                            if self.distributed:
                                # 汇总所有 worker 的评估结果。
                                eval_dict =  torch.tensor([eval_dict['avg_loss'], eval_dict['avg_acc'], *eval_dict['cls_acc']], device=self.device)
                                dist.reduce(eval_dict, dst=0, op=dist.ReduceOp.SUM)
                                eval_dict = eval_dict.cpu().numpy()
                                eval_dict = {'avg_loss': eval_dict[0]/self.world_size, 'avg_acc': eval_dict[1]/self.world_size, 'cls_acc': eval_dict[2:]/self.world_size}
                            if self.is_main_process():
                                eval_results["test_acc"].append(eval_dict['avg_acc'])
                                eval_results["avg_acc"].append(eval_dict['cls_acc'])
                                eval_results["data_cnt"].append(num_eval)
                                self.report_test(num_eval, eval_dict["avg_loss"], eval_dict['avg_acc'])
                            num_eval += self.eval_period

                    sys.stdout.flush()

                test_sampler = OnlineTestSampler(self.test_dataset, self.exposed_classes)
                test_dataloader = DataLoader(self.test_dataset, batch_size=self.batchsize*2, sampler=test_sampler, num_workers=self.n_worker)
                # 每个 epoch 结束时做一次评估快照，在这里也基本等价于任务末评估。
                eval_dict = self.online_evaluate(test_dataloader, task_id=task_id, end=True)

            # 任务结束钩子：给具体方法做清理或状态切换。
            self.online_after_task(task_id)

            if self.distributed:
                eval_dict =  torch.tensor([eval_dict['avg_loss'], eval_dict['avg_acc'], *eval_dict['cls_acc']], device=self.device)
                dist.reduce(eval_dict, dst=0, op=dist.ReduceOp.SUM)
                eval_dict = eval_dict.cpu().numpy()
                eval_dict = {'avg_loss': eval_dict[0]/self.world_size, 'avg_acc': eval_dict[1]/self.world_size, 'cls_acc': eval_dict[2:]/self.world_size}
            task_acc = eval_dict['avg_acc']

            if self.is_main_process():
                self._append_progress_record({
                    "event": "task_end",
                    "task_id": task_id,
                    "sample_num": int(samples_cnt),
                    "progress_percent": self._training_progress_percent(samples_cnt),
                    "task_acc": float(task_acc),
                    "elapsed_sec": int(time.time() - self.start_time),
                })

            logger.info("[2-4] Update the information for the current task")
            task_records["task_acc"].append(task_acc)
            task_records["cls_acc"].append(eval_dict["cls_acc"])

            logger.info("[2-5] Report task result")
            logger.info(task_records['task_acc'])

        # ================== 汇总结果 ===================
        if self.is_main_process():

            # 准确率（A）
            # A_auc：周期性在线评估准确率的平均值。
            # A_avg：每个任务结束时准确率的平均值。
            # A_last：最后一个任务结束时的准确率。
            A_auc = float(np.mean(eval_results["test_acc"])) if len(eval_results["test_acc"]) > 0 else None
            A_avg = float(np.mean(task_records["task_acc"]))
            A_last = float(task_records["task_acc"][self.n_tasks - 1])

            # 遗忘（F）
            # F_last 计算的是每个类别从历史最好成绩到最终成绩的下降幅度。
            cls_acc = np.array(task_records["cls_acc"])
            acc_diff = []
            if self.n_tasks > 1:
                for j in range(self.n_classes):
                    if np.max(cls_acc[:-1, j]) > 0:
                        acc_diff.append(np.max(cls_acc[:-1, j]) - cls_acc[-1, j])
                F_last = float(np.mean(acc_diff)) if len(acc_diff) > 0 else -999
            else:
                F_last = -999

            # 反向迁移（BWT），按类别计算：
            # 最终准确率 - 该类第一次学到时的准确率（最后一个任务之前第一次非零准确率）
            if self.n_tasks > 1:
                bwt_vals = []
                for j in range(self.n_classes):
                    per_cls_prev = cls_acc[:-1, j]
                    seen_indices = np.where(per_cls_prev > 0)[0]
                    if len(seen_indices) == 0:
                        continue
                    first_acc = per_cls_prev[seen_indices[0]]
                    last_acc = cls_acc[-1, j]
                    bwt_vals.append(last_acc - first_acc)
                if len(bwt_vals) > 0:
                    BWT_last = float(np.mean(bwt_vals))
                else:
                    BWT_last = -999
            else:
                BWT_last = -999

            logger.info("======== Summary =======")
            logger.info(self.note)
            logger.info(f"A_auc {A_auc} | A_avg {A_avg} | A_last {A_last} | F_last {F_last}")
            logger.info(f"BWT_last {BWT_last}")
            logger.info("="*24)
            logger.info(eval_results['test_acc'])

            np.save(f"{self.log_dir}/seed_{self.rnd_seed}.npy", task_records["task_acc"])

            if self.eval_period != np.inf:
                np.save(f'{self.log_dir}/seed_{self.rnd_seed}_eval.npy', eval_results['test_acc'])
                np.save(f'{self.log_dir}/seed_{self.rnd_seed}_eval_time.npy', eval_results['data_cnt'])

            # 可选的事后 expert 表征分析（例如 FlyPrompt / DualPrompt / MVP）。
            if getattr(self, "analysis_expert_similarity", False):
                if hasattr(self, "analyze_expert_features"):
                    logger.info("[Post] Running expert feature similarity / CKA analysis ...")
                    try:
                        self.analyze_expert_features()
                    except Exception as e:
                        logger.exception("[Post] Expert feature analysis failed: %s", e)
                else:
                    logger.info(
                        "[Post] analysis_expert_similarity=True but method has no "
                        "analyze_expert_features; skipping expert analysis."
                    )

            self._save_final_checkpoint()
            self._append_progress_record({
                "event": "run_end",
                "task_id": self.n_tasks - 1,
                "sample_num": int(samples_cnt),
                "progress_percent": 100.0,
                "A_auc": A_auc,
                "A_avg": float(A_avg),
                "A_last": float(A_last),
                "F_last": float(F_last),
                "BWT_last": float(BWT_last),
                "elapsed_sec": int(time.time() - self.start_time),
            })

    def profile_worker(self, gpu) -> None:
        # 轻量级执行路径，用来快速跑一个小迭代做 profiling。
        # ============ 小型实验初始化 ============
        self.gpu    = gpu % self.ngpus_per_nodes
        self.device = torch.device(self.gpu)
        if self.distributed:
            self.local_rank = self.gpu
            if 'SLURM_PROCID' in os.environ.keys():
                self.rank = int(os.environ['SLURM_PROCID']) * self.ngpus_per_nodes + self.gpu
                logger.info(f"| Init Process group {os.environ['SLURM_PROCID']} : {self.local_rank}")
            else :
                self.rank = self.gpu
                logger.info(f"| Init Process group 0 : {self.local_rank}")
            if 'MASTER_ADDR' not in os.environ.keys():
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '12702'
            torch.cuda.set_device(self.gpu)
            time.sleep(self.rank * 0.1) # 防止端口冲突
            dist.init_process_group(backend=self.dist_backend, init_method=self.dist_url,
                                    world_size=self.world_size, rank=self.rank)
            torch.distributed.barrier()
            self.setup_for_distributed(self.is_main_process())
        else:
            pass

        if self.rnd_seed is not None:
            random.seed(self.rnd_seed)
            np.random.seed(self.rnd_seed)
            torch.manual_seed(self.rnd_seed)
            torch.cuda.manual_seed(self.rnd_seed)
            torch.cuda.manual_seed_all(self.rnd_seed) # 多 GPU 时也要同步设置
            cudnn.deterministic = True
        cudnn.benchmark = False

        self.setup_distributed_dataset()
        self.total_samples = len(self.train_dataset)
        self._init_internal_step_scheduler()

        self.setup_distributed_model()

        samples_cnt = 0
        self.train_sampler.set_task(0)
        self.online_before_task(0)
        for i, (images, labels, idx) in enumerate(self.train_dataloader):
            samples_cnt += images.size(0) * self.world_size
            loss, acc = self.online_step(images, labels, idx)
            self.report_training(samples_cnt, loss, acc)
            break
        self.online_after_task(0)

    def add_new_class(self, class_name):
        # 用新观察到的真实标签更新 exposed_classes。
        # 使用场景：在线流式训练中，一个 batch 里第一次出现新类别时，
        # 需要把它加入“已见类别集合”，否则后面算 loss / eval 时会把它挡住。
        # 在分布式模式下，会把所有 worker 的类别集合收集并合并。
        exposed_classes = []
        new = []
        for label in class_name:
            if label.item() not in self.exposed_classes:
                self.exposed_classes.append(label.item())
                new.append(label.item())
        if self.distributed:
            exposed_classes = torch.cat(self.all_gather(torch.tensor(self.exposed_classes, device=self.device))).cpu().tolist()
            self.exposed_classes = []
            for cls in exposed_classes:
                if cls not in self.exposed_classes:
                    self.exposed_classes.append(cls)
        # 具体方法里会把 exposed_classes 映射到 [0, len(exposed_classes)-1]，
        # 因此这里要把对应前缀的 logits 解开 mask。
        self.mask[:len(self.exposed_classes)] = 0

        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

    def online_step(self, sample, samples_cnt):
        # 钩子：对一个进入的 batch 执行一次在线优化。
        # 使用场景：子类（如 FlyPrompt）会在这里写“这个 batch 怎么训练”。
        # 例如：先做类别映射，再 forward，再 backward，再更新优化器。
        raise NotImplementedError()

    def online_before_task(self, task_id):
        # 钩子：benchmark 任务开始前的具体方法准备工作。
        # 使用场景：如果某方法需要在任务开始时重置缓冲区、切换 prompt、
        # 或者初始化某些 task 级状态，就在子类里实现它。
        raise NotImplementedError()

    def online_after_task(self, task_id):
        # 钩子：benchmark 任务结束后的具体方法状态切换。
        # 使用场景：一个任务学完后，可能要冻结旧状态、保存统计量，
        # 或者像 FlyPrompt 那样把内部 task_id 前进一格。
        raise NotImplementedError()

    def online_evaluate(self, test_loader, samples_cnt, task_id=None, end=False):
        # 钩子：具体方法自己的评估逻辑。
        # 使用场景：不同方法可能有不同评估方式，
        # 比如 FlyPrompt 会先路由到 expert，再把多个头的输出融合。
        raise NotImplementedError()

    def update_schedule(self, reset=False):
        # 统一的学习率调度更新工具，供各方法在 optimizer.step() 后调用。
        # 使用场景：一个 batch 反向传播完成后，通常会调用一次。
        # 如果某些方法在任务切换时要重置学习率，也可以传 reset=True。
        if reset:
            self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr
        else:
            self.scheduler.step()

    def is_dist_avail_and_initialized(self):
        if not dist.is_available():
            return False
        if not dist.is_initialized():
            return False
        return True

    def get_world_size(self):
        if not self.is_dist_avail_and_initialized():
            return 1
        return dist.get_world_size()

    def get_rank(self):
        if not self.is_dist_avail_and_initialized():
            return 0
        return dist.get_rank()

    def is_main_process(self):
        return self.get_rank() == 0

    def setup_for_distributed(self, is_master):
        """
        当不是主进程时，这个函数会屏蔽 print 和日志输出。
        """
        import builtins as __builtin__
        builtin_print = __builtin__.print

        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if is_master or force:
                builtin_print(*args, **kwargs)
        __builtin__.print = print

        class MasterOnlyFilter(logging.Filter):
            def __init__(self, is_master):
                super().__init__()
                self.is_master = is_master

            def filter(self, record):
                return self.is_master or record.levelno < logging.INFO

        for h in logging.getLogger().handlers:
            h.addFilter(MasterOnlyFilter(is_master))

    def report_training(self, sample_num, train_loss, train_acc):
        # 使用场景：训练过程中每隔一段样本数打印一次。
        # 例如你在看日志时，会看到“当前处理了多少样本、loss、acc、lr”。
        progress = self._training_progress_percent(sample_num)
        logger.info(
            f"Train | Sample # {sample_num} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"progress {progress:.2f}% | "
            f"lr {self.optimizer.param_groups[0]['lr']:.6f} | "
            f"Num_Classes {len(self.exposed_classes)} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples*self.num_epochs-sample_num) / sample_num))}"
        )
        self._append_progress_record({
            "event": "train",
            "task_id": self.current_task_id,
            "sample_num": int(sample_num),
            "progress_percent": progress,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "lr": float(self.optimizer.param_groups[0]['lr']),
            "elapsed_sec": int(time.time() - self.start_time),
        })

    def report_test(self, sample_num, avg_loss, avg_acc):
        # 使用场景：在线评估触发时打印，例如每累计到 eval_period 个样本。
        progress = self._training_progress_percent(sample_num)
        logger.info(
            f"Test | Sample # {sample_num} | test_loss {avg_loss:.4f} | test_acc {avg_acc:.4f} | "
        )
        self._append_progress_record({
            "event": "test",
            "task_id": self.current_task_id,
            "sample_num": int(sample_num),
            "progress_percent": progress,
            "test_loss": float(avg_loss),
            "test_acc": float(avg_acc),
            "elapsed_sec": int(time.time() - self.start_time),
        })

    def _training_progress_percent(self, sample_num: int) -> float:
        total_steps = max(1, int(self.total_samples * self.num_epochs))
        return min(100.0, max(0.0, (sample_num / total_steps) * 100.0))

    def _append_progress_record(self, record: dict) -> None:
        if not self.is_main_process():
            return
        record = dict(record)
        record.setdefault("note", self.note)
        record.setdefault("seed", getattr(self, "rnd_seed", None))
        record.setdefault("method", getattr(self, "method", None))
        record.setdefault("backbone", getattr(self, "backbone", None))
        record.setdefault("num_tasks", getattr(self, "n_tasks", None))
        try:
            with open(self.progress_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("Failed to append training progress record to %s", self.progress_log_path)

    def _save_final_checkpoint(self) -> None:
        if not self.is_main_process():
            return
        model = getattr(self, "model_without_ddp", None)
        if model is None:
            model = getattr(self, "model", None)
        if model is None:
            logger.warning("No model available for final checkpoint save; skipping.")
            return

        optimizer = getattr(self, "optimizer", None)
        scheduler = getattr(self, "scheduler", None)
        checkpoint = {
            "seed": getattr(self, "rnd_seed", None),
            "note": getattr(self, "note", None),
            "method": getattr(self, "method", None),
            "backbone": getattr(self, "backbone", None),
            "num_epochs": getattr(self, "num_epochs", None),
            "n_tasks": getattr(self, "n_tasks", None),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None and hasattr(scheduler, "state_dict") else None,
            "exposed_classes": getattr(self, "exposed_classes", None),
        }
        torch.save(checkpoint, self.final_checkpoint_path)
        logger.info("Saved final checkpoint to %s", self.final_checkpoint_path)

    def _interpret_pred(self, y, pred):
        # 返回每个类别的样本数和正确数，用于计算类别级准确率。
        # 使用场景：评估时想知道“每个类分别对了多少”，
        # 而不只是整体 accuracy。
        # 例子：如果一个 batch 里有 3 张 cat、2 张 dog，这里会分别统计它们的总数和预测正确数。
        ret_num_data = torch.zeros(self.n_classes)
        ret_corrects = torch.zeros(self.n_classes)

        xlabel_cls, xlabel_cnt = y.unique(return_counts=True)
        for cls_idx, cnt in zip(xlabel_cls, xlabel_cnt):
            ret_num_data[cls_idx] = cnt

        correct_xlabel = y.masked_select(y == pred)
        correct_cls, correct_cnt = correct_xlabel.unique(return_counts=True)
        for cls_idx, cnt in zip(correct_cls, correct_cnt):
            ret_corrects[cls_idx] = cnt

        return ret_num_data, ret_corrects

    def reset_opt(self):
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)

    def all_gather(self, item):
        # 先把不同 worker 的可变长 tensor 补齐到同样长度，再进行 gather，
        # 最后把补的部分裁掉。
        # 使用场景：收集分布式训练下各 worker 的类别集合时，
        # 每个 worker 看到的新类别数量可能不一样，不能直接拼。
        local_size = torch.tensor(item.size(0), device=self.device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                dist.gather(local_size, all_sizes, dst=i)
            else:
                dist.gather(local_size, dst=i)
        max_size = max(all_sizes)

        size_diff = max_size.item() - local_size.item()
        if size_diff:
            padding = torch.zeros(size_diff, device=self.device, dtype=item.dtype)
            item = torch.cat((item, padding))

        all_qs_padded = [torch.zeros_like(item) for _ in range(dist.get_world_size())]

        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                dist.gather(item, all_qs_padded, dst=i)
            else:
                dist.gather(item, dst=i)

        all_qs = []
        for q, size in zip(all_qs_padded, all_sizes):
            all_qs.append(q[:size])
        return all_qs

    def train_data_config(self, n_task, train_dataset, train_sampler):
        for t_i in range(n_task):
            train_sampler.set_task(t_i)
            train_dataloader = DataLoader(train_dataset,batch_size=self.batchsize,sampler=train_sampler,num_workers=4)
            data_info={}
            for i,data in enumerate(train_dataloader):
                _,label = data
                label = label.to(self.device)
                for b in range(len(label)):
                    if 'Class_'+str(label[b].item()) in data_info.keys():
                        data_info['Class_'+str(label[b].item())] += 1
                    else:
                        data_info['Class_'+str(label[b].item())] = 1
            logger.info(f"[Train] Task{t_i} Data Info")
            logger.info(data_info)
            convert_data_info = self.convert_class_label(data_info)
            np.save(f"{self.log_dir}/seed_{self.rnd_seed}_task{t_i}_train_data.npy", convert_data_info)
            logger.info(f"[Train] Task{t_i} Converted Data Info")
            logger.info(convert_data_info)
            logger.info("")

    def test_data_config(self, test_dataloader, task_id):
        data_info={}
        for i,data in enumerate(test_dataloader):
            _,label = data
            label = label.to(self.device)
            for b in range(len(label)):
                if 'Class_'+str(label[b].item()) in data_info.keys():
                    data_info['Class_'+str(label[b].item())]+=1
                else:
                    data_info['Class_'+str(label[b].item())]=1
        logger.info("[Test] Exposed Classes:")
        logger.info(self.exposed_classes)
        logger.info(f"[Test] Task {task_id} Data Info")
        logger.info(data_info)
        logger.info(f"[Test] Task{task_id} Converted Data Info")
        convert_data_info = self.convert_class_label(data_info)
        logger.info(convert_data_info)
        logger.info("")

    def convert_class_label(self,data_info):
        #* self.class_list => 原始类别标签
        self.class_list = self.train_dataset.classes
        for key in list(data_info.keys()):
            old_key= int(key[6:])
            data_info[self.class_list[old_key]] = data_info.pop(key)
        return data_info

    def current_task_data(self,train_loader):
        data_info={}
        for i,data in enumerate(train_loader):
            _,label = data
            for b in range(label.shape[0]):
                if 'Class_'+str(label[b].item()) in data_info.keys():
                    data_info['Class_'+str(label[b].item())] +=1
                else:
                    data_info['Class_'+str(label[b].item())] =1
        logger.info("[Current Task] Data Info")
        logger.info(data_info)
        logger.info("[Current Task] Converted Data Info")
        convert_data_info = self.convert_class_label(data_info)
        logger.info(convert_data_info)
        logger.info("")
