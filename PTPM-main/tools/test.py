import _init_path
import argparse
import datetime
import glob
import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
import re
import time
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils


def parse_config():
    """
    解析命令行参数并配置实验参数
    
    Returns:
        args: 解析后的命令行参数
        cfg: 更新后的配置对象
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='arg parser')
    
    # 添加配置文件参数
    parser.add_argument('--cfg_file', type=str, default=None, help='指定训练配置文件路径')

    # 添加训练相关参数
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='训练的batch大小')
    parser.add_argument('--workers', type=int, default=4, help='数据加载器的worker数量')
    parser.add_argument('--extra_tag', type=str, default='default', help='本次实验的额外标签')
    parser.add_argument('--ckpt', type=str, default=None, help='起始检查点路径')
    parser.add_argument('--pretrained_model', type=str, default=None, help='预训练模型路径')
    
    # 添加分布式训练参数
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='分布式训练启动器类型')
    parser.add_argument('--tcp_port', type=int, default=18888, help='分布式训练的TCP端口')
    parser.add_argument('--local_rank', type=int, default=0, help='分布式训练的本地rank')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='设置额外的配置键值对')

    # 添加评估相关参数
    parser.add_argument('--max_waiting_mins', type=int, default=30, help='最大等待时间(分钟)')
    parser.add_argument('--start_epoch', type=int, default=0, help='起始epoch')
    parser.add_argument('--eval_tag', type=str, default='default', help='评估标签')
    parser.add_argument('--eval_all', action='store_true', default=False, help='是否评估所有检查点')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='指定要评估的检查点目录')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='是否保存结果到文件')
    parser.add_argument('--infer_time', action='store_true', default=False, help='是否计算推理延迟')

    # 解析参数
    args = parser.parse_args()

    # 从YAML文件加载配置
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # 移除'cfgs'和'xxxx.yaml'

    # 设置随机种子
    np.random.seed(1024)

    # 如果有额外配置,更新配置
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False, ceph_output_path=None):
    # 检查checkpoint文件是否存在
    ckpt_file = args.ckpt.replace('../output', ceph_output_path) if ceph_output_path else args.ckpt
    logger.info(f'Loading checkpoint from: {ckpt_file}')
    
    if not os.path.exists(ckpt_file):
        logger.error(f'Checkpoint file not found: {ckpt_file}')
        return
    
    try:
        if ceph_output_path:
            assert args.ckpt.startswith('../output')
            model.load_params_from_file(filename=args.ckpt.replace('../output', ceph_output_path), logger=logger, to_cpu=dist_test, 
                                    pre_trained_path=args.pretrained_model)
        else:
            model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test, 
                                    pre_trained_path=args.pretrained_model)
    except Exception as e:
        logger.error(f'Error loading checkpoint: {str(e)}')
        return
        
    # 检查模型是否成功加载到GPU
    logger.info(f'Moving model to GPU...')
    model.cuda()
    logger.info(f'Model device after cuda(): {next(model.parameters()).device}')
    
    # 检查GPU是否可用
    logger.info(f'CUDA available: {torch.cuda.is_available()}')
    logger.info(f'Current GPU device: {torch.cuda.current_device()}')
    logger.info(f'GPU device name: {torch.cuda.get_device_name(torch.cuda.current_device())}')
    
    # start evaluation
    logger.info('Starting evaluation...')
    eval_utils.eval_one_epoch(
        cfg, args, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir
    )


def get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args):
    """
    获取未评估的检查点文件
    
    Args:
        ckpt_dir: 检查点文件目录
        ckpt_record_file: 记录已评估检查点的文件
        args: 参数配置
        
    Returns:
        epoch_id: 检查点的轮次ID,如果没有找到则返回-1
        ckpt_path: 检查点文件路径,如果没有找到则返回None
    """
    # 处理S3存储的情况
    if 's3://' in str(ckpt_dir):
        # 获取S3目录下所有检查点文件
        ckpt_list = petrel_client.list_dir_one_depth(ckpt_dir)
        # 提取检查点文件的轮次ID
        epoch_list = [int(os.path.basename(ckpt_file)[17:-4]) for ckpt_file in ckpt_list 
                     if ckpt_file.startswith('checkpoint') and ckpt_file.endswith('.pth')]
        epoch_list.sort()
        # 读取已评估的检查点列表
        evaluated_ckpt_list = [float(x.strip()) for x in open(ckpt_record_file, 'r').readlines()]

        # 查找未评估且符合起始轮次要求的检查点
        for epoch_id in epoch_list:
            if float(epoch_id) not in evaluated_ckpt_list and int(float(epoch_id)) >= args.start_epoch:
                return epoch_id, os.path.join(ckpt_dir, "checkpoint_epoch_%d.pth" % epoch_id)
        return -1, None

    # 处理本地存储的情况
    else:
        # 获取本地目录下所有检查点文件
        ckpt_list = glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))
        # 按修改时间排序
        ckpt_list.sort(key=os.path.getmtime)
        # 读取已评估的检查点列表
        evaluated_ckpt_list = [float(x.strip()) for x in open(ckpt_record_file, 'r').readlines()]

        # 遍历检查点文件
        for cur_ckpt in ckpt_list:
            # 提取轮次ID
            num_list = re.findall('checkpoint_epoch_(.*).pth', cur_ckpt)
            if num_list.__len__() == 0:
                continue

            epoch_id = num_list[-1]
            # 跳过优化器检查点
            if 'optim' in epoch_id:
                continue
            # 查找未评估且符合起始轮次要求的检查点
            if float(epoch_id) not in evaluated_ckpt_list and int(float(epoch_id)) >= args.start_epoch:
                return epoch_id, cur_ckpt
        return -1, None


def repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=False, save_result=True):
    """
    重复评估检查点的函数
    
    Args:
        model: 待评估的模型
        test_loader: 测试数据加载器
        args: 参数配置
        eval_output_dir: 评估结果输出目录
        logger: 日志记录器
        ckpt_dir: 检查点目录
        dist_test: 是否使用分布式测试,默认False
        save_result: 是否保存结果,默认True
        
    功能说明:
    1. 创建评估记录文件,用于记录已评估的检查点
    2. 如果是主进程(rank=0),创建tensorboard日志
    3. 循环检查是否有未评估的检查点:
       - 获取未评估的检查点
       - 如果没有找到或轮次小于起始轮次,等待30秒后继续检查
       - 如果等待时间超过最大等待时间且不是首次评估,则退出
    4. 加载检查点参数并开始评估:
       - 将模型加载到GPU
       - 执行一轮评估
       - 记录tensorboard日志
       - 将已评估的检查点记录到文件
    """
    # 创建评估记录文件
    ckpt_record_file = eval_output_dir / ('eval_list_%s.txt' % cfg.DATA_CONFIG.DATA_SPLIT['test'])
    with open(ckpt_record_file, 'a'):
        pass

    # 创建tensorboard日志(仅主进程)
    if cfg.LOCAL_RANK == 0:
        tb_log = SummaryWriter(log_dir=str(eval_output_dir / ('tensorboard_%s' % cfg.DATA_CONFIG.DATA_SPLIT['test'])))
    total_time = 0
    first_eval = True

    while True:
        # 检查是否有未评估的检查点
        cur_epoch_id, cur_ckpt = get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args)
        if cur_epoch_id == -1 or int(float(cur_epoch_id)) < args.start_epoch:
            wait_second = 30
            if cfg.LOCAL_RANK == 0:
                print('Wait %s seconds for next check (progress: %.1f / %d minutes): %s \r'
                      % (wait_second, total_time * 1.0 / 60, args.max_waiting_mins, ckpt_dir), end='', flush=True)
            time.sleep(wait_second)
            total_time += 30
            if total_time > args.max_waiting_mins * 60 and (first_eval is False):
                break
            continue

        total_time = 0
        first_eval = False

        # 加载检查点参数
        model.load_params_from_file(filename=cur_ckpt, logger=logger, to_cpu=dist_test)
        model.cuda()

        # 开始评估
        cur_result_dir = eval_output_dir / ('epoch_%s' % cur_epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
        tb_dict = eval_utils.eval_one_epoch(
            cfg, args, model, test_loader, cur_epoch_id, logger, dist_test=dist_test,
            result_dir=cur_result_dir,
            save_result=save_result
        )

        # 记录tensorboard日志
        if cfg.LOCAL_RANK == 0:
            for key, val in tb_dict.items():
                tb_log.add_scalar(key, val, cur_epoch_id)

        # 记录已评估的检查点
        with open(ckpt_record_file, 'a') as f:
            print('%s' % cur_epoch_id, file=f)
        logger.info('Epoch %s has been evaluated' % cur_epoch_id)


def main():
    """
    主函数,用于执行模型评估。主要功能包括:
    1. 设置GPU和环境变量
    2. 解析配置参数
    3. 设置分布式训练
    4. 准备输出目录和日志
    5. 构建数据加载器
    6. 加载模型
    7. 执行评估
    """
    # 设置使用的GPU设备
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    # 解析命令行参数和配置文件
    args, cfg = parse_config()

    # 如果需要推理时间统计,设置CUDA同步执行
    if args.infer_time:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # 设置分布式训练相关参数
    if args.launcher == 'none':
        dist_test = False  # 不使用分布式
        total_gpus = 1
    else:
        # 初始化分布式环境
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    # 设置batch size
    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size应该能被GPU数量整除'
        args.batch_size = args.batch_size // total_gpus

    # 创建输出目录
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    # 设置评估输出目录
    if not args.eval_all:
        # 从checkpoint文件名提取epoch数
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    else:
        eval_output_dir = eval_output_dir / 'eval_all_default'

    # 添加额外的评估标签
    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    # 创建日志目录和日志记录器
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # 记录基本信息到日志
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    # 记录配置信息
    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    # 设置checkpoint目录
    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'

    # 构建数据加载器
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    # 获取CEPH输出路径(如果有)
    ceph_output_path = cfg.DATA_CONFIG.get('CEPH_OUTPUT_PATH', None)

    # 构建网络模型
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    
    # 执行评估
    with torch.no_grad():
        if args.eval_all:
            # 评估所有checkpoints
            repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=dist_test)
        else:
            try:
                # 评估单个checkpoint
                eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=dist_test, ceph_output_path=ceph_output_path)
            except IndexError as e:
                logger.error("评估过程中出错: %s" % str(e))
                logger.info("此错误可能由空的预测结果导致。请检查模型checkpoint和数据。")
                return
            except Exception as e:
                logger.error("评估过程中发生意外错误: %s" % str(e))
                return


if __name__ == '__main__':
    main()
