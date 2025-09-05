import torch
import math

def linear_scheduler(optimizer, total_steps):
    """LinearLR: 학습률을 선형적으로 감소"""
    return torch.optim.lr_scheduler.LinearLR(
        optimizer,
        total_iters=total_steps,
        last_epoch=-1
    )

def cycle_scheduler(optimizer, total_steps, max_lr, num_epochs):
    """OneCycleLR: warmup + decay 사이클 학습률"""
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        final_div_factor=1000,
        last_epoch=-1,
        pct_start=2 / num_epochs
    )

def get_scheduler(args, optimizer, num_train):
    """학습률 스케줄러 선택자"""
    # 한 번에 돌릴 수 있는 데이터 개수: batch개수 * gpu개수
    global_batch_size = args.batch_size * 1
    # global_batch_size = args.batch_size * world_size
    # epoch개수 * (전체 데이터 개수 // 한 번에 돌릴 수 있는 데이터 개수)
    total_steps = int(args.num_epochs * math.ceil(num_train / global_batch_size)) # 작은 데이터라도 남아 있으면 1개의 batch로 처리하도록 step 수를 올려줌

    if args.scheduler_type == 'linear':
        return linear_scheduler(optimizer, total_steps)
    if args.scheduler_type == 'cycle':
        return cycle_scheduler(optimizer, total_steps, args.max_lr, args.num_epochs)
