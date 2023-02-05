import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

import commons
import utils
from data_utils import (
  TextAudioLoader,
  TextAudioCollate,
  DistributedBucketSampler
)
from models import (
  SynthesizerTrn,
  MultiPeriodDiscriminator,
)
from losses import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text import create_symbols_manager

import time

torch.backends.cudnn.benchmark = True
global_step = 0


def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."

  n_gpus = torch.cuda.device_count()
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '8000'

  hps = utils.get_hparams()
  if 'language' in hps.data:
    hps.symbols_manager = create_symbols_manager(hps.data.language)
  else:
    hps.symbols_manager = create_symbols_manager('default')
    
  mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
  global global_step
  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  # https://github.com/ray-project/ray_lightning/issues/13
  from sys import platform
  if platform == "win32":
    backend = 'gloo'
  else:
    backend = 'nccl'
  dist.init_process_group(backend=backend, init_method='env://', world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)

  train_dataset = TextAudioLoader(hps.data.training_files, hps.data, hps.symbols_manager)
  collate_fn = TextAudioCollate()
  if hps.data_loader.use_train_sampler:
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32,300,400,500,600,700,800,900,1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True)
    train_loader = DataLoader(train_dataset, num_workers=hps.data_loader.num_workers, shuffle=False, pin_memory=True,
        collate_fn=collate_fn, batch_sampler=train_sampler)
  else:
    train_loader = DataLoader(train_dataset, num_workers=hps.data_loader.num_workers, shuffle=False,
        batch_size=hps.train.batch_size, pin_memory=True, 
        drop_last=False, collate_fn=collate_fn)
  if rank == 0:
    eval_dataset = TextAudioLoader(hps.data.validation_files, hps.data, hps.symbols_manager)
    eval_loader = DataLoader(eval_dataset, num_workers=hps.data_loader.num_workers, shuffle=False,
        batch_size=hps.train.batch_size, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)

  net_g = SynthesizerTrn(
      len(hps.symbols_manager.symbols),
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      **hps.model).cuda(rank)
  net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
  optim_g = torch.optim.AdamW(
      net_g.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  optim_d = torch.optim.AdamW(
      net_d.parameters(),
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  net_g = DDP(net_g, device_ids=[rank])
  net_d = DDP(net_d, device_ids=[rank])

  use_pretrained_weights = False
  overwrite_lr = False
  try:
    # Pretrained_model_name means the pretrained weights, bad naming
    utils.load_checkpoint(utils.latest_checkpoint_path(
      "pretrained_models", 
      f"G_{hps.checkpoints.pretrained_model_name}*.pth"
      ), 
      net_g, 
      optim_g
    )
    utils.load_checkpoint(utils.latest_checkpoint_path(
      "pretrained_models", f"D_{hps.checkpoints.pretrained_model_name}*.pth"
      ), 
      net_d, 
      optim_d
    )
    use_pretrained_weights = True
    overwrite_lr = True
  except:
    print("No pretrained weights to load")
    use_pretrained_weights = False
    overwrite_lr = False

  try:
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
    global_step = (epoch_str - 1) * len(train_loader)
    overwrite_lr = False
  except:
    epoch_str = 1
    global_step = 0

  if ('force_overwrite_lr' in hps.train and
      hps.train.force_overwrite_lr):
    overwrite_lr = True

  def set_lr(optim, lr):
    for g in optim.param_groups:
      g['lr'] = lr

  if overwrite_lr:
    print("Overwrite learning rates of optimizers")
    set_lr(optim_g, hps.train.learning_rate)
    set_lr(optim_d, hps.train.learning_rate)

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

  scaler = GradScaler(enabled=hps.train.fp16_run)

  for epoch in range(epoch_str, hps.train.epochs + 1):
    start = time.perf_counter()
    if rank==0:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, eval_loader], logger, [writer, writer_eval])
    else:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, None], None, None)
    print(f"This epoch takes {time.perf_counter() - start} seconds")
    scheduler_g.step()
    scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
  net_g, net_d = nets
  optim_g, optim_d = optims
  scheduler_g, scheduler_d = schedulers
  train_loader, eval_loader = loaders
  if writers is not None:
    writer, writer_eval = writers

  if hps.data_loader.use_train_sampler:
    train_loader.batch_sampler.set_epoch(epoch)
  global global_step

  net_g.train()
  net_d.train()

  # TODO: later should make it configurable
  if rank == 0:
    loss_gen_all_min = 35.0
    g_min_checkpoint_index = 0

  for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(train_loader):
    x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
    spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
    y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)

    with autocast(enabled=hps.train.fp16_run):
      y_hat, l_length, attn, ids_slice, x_mask, z_mask,\
      (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, x_lengths, spec, spec_lengths)

      mel = spec_to_mel_torch(
          spec, 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate,
          hps.data.mel_fmin, 
          hps.data.mel_fmax)
      y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
      y_hat_mel = mel_spectrogram_torch(
          y_hat.squeeze(1), 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate, 
          hps.data.hop_length, 
          hps.data.win_length, 
          hps.data.mel_fmin, 
          hps.data.mel_fmax
      )

      y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size) # slice 

      # Discriminator
      y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
      with autocast(enabled=False):
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc
    optim_d.zero_grad()
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    scaler.step(optim_d)

    with autocast(enabled=hps.train.fp16_run):
      # Generator
      y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
      with autocast(enabled=False):
        loss_dur = torch.sum(l_length.float())
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
    optim_g.zero_grad()
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(optim_g)
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    scaler.step(optim_g)
    scaler.update()

    if rank==0 and global_step % hps.train.log_interval == 0:
      if loss_gen_all < loss_gen_all_min:
        loss_gen_all_min = loss_gen_all
        if g_min_checkpoint_index > 0:
          g_min_checkpoint_old_path = os.path.join(hps.model_dir, f"G_min_{g_min_checkpoint_index}.pth")
          g_d_min_checkpoint_old_path = os.path.join(hps.model_dir, f"G_D_min_{g_min_checkpoint_index}.pth")
          # https://stackoverflow.com/questions/53028607/how-to-remove-the-file-from-trash-in-drive-in-colab
          open(g_min_checkpoint_old_path, 'w').close() # Overwrite and make the file blank
          open(g_d_min_checkpoint_old_path, 'w').close()
          os.remove(f"{g_min_checkpoint_old_path}")
          os.remove(f"{g_d_min_checkpoint_old_path}")
        
        g_min_checkpoint_index += 1
        g_min_checkpoint_path = os.path.join(hps.model_dir, f"G_min_{g_min_checkpoint_index}.pth")
        g_d_min_checkpoint_path = os.path.join(hps.model_dir, f"G_D_min_{g_min_checkpoint_index}.pth")
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, g_min_checkpoint_path)
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, g_d_min_checkpoint_path)

    if rank==0:
      if global_step % hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']
        print(f"The current learning rate is: {lr}")
        losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info([x.item() for x in losses] + [global_step, lr])
        
        scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
        scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/dur": loss_dur, "loss/g/kl": loss_kl})

        scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
        scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
        scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
        image_dict = { 
            "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()), 
            "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
            "all/attn": utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy())
        }
        utils.summarize(
          writer=writer,
          global_step=global_step, 
          images=image_dict,
          scalars=scalar_dict)

      if global_step % hps.train.eval_interval == 0:
        evaluate(hps, net_g, eval_loader, writer_eval)

        should_auto_delete_old_checkpoints = False
        if hasattr(hps, 'checkpoints'):
          if hps.checkpoints.auto_delete_old_checkpoints:
            should_auto_delete_old_checkpoints = True
        else:
          logger.info(("The 'checkpoints' config option hasn't been specified!"))

        g_checkpoint_path = os.path.join(hps.model_dir, "G_{}_{}.pth".format(hps.model_name, global_step))
        d_checkpoint_path = os.path.join(hps.model_dir, "D_{}_{}.pth".format(hps.model_name, global_step))
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, g_checkpoint_path)
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, d_checkpoint_path)

        # Only keep a number of latest checkpoints to save disc space
        if should_auto_delete_old_checkpoints:
          num_checkpoints = utils.number_of_checkpoints(hps.model_dir, "G_*.pth")
          num_checkpoints_to_keep = hps.checkpoints.num_checkpoints_to_keep
          g_oldest_checkpoint_path = utils.oldest_checkpoint_path(hps.model_dir, "G_*.pth")
          d_oldest_checkpoint_path = utils.oldest_checkpoint_path(hps.model_dir, "D_*.pth")
          if hps.checkpoints.replace_old_checkpoints_mode:
            # Deprecated: this method won't work on Google colab
            if num_checkpoints >= num_checkpoints_to_keep:
              # First save the latest checkpoint into the oldest checkpoint file.
              # Then, rename the newly saved file to latest checkpoint name.
              utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, g_oldest_checkpoint_path, False)
              logger.info("Saving model and optimizer state at iteration {} to {}".format(epoch, g_checkpoint_path))
              os.rename(g_oldest_checkpoint_path, g_checkpoint_path)

              logger.info("Saving model and optimizer state at iteration {} to {}".format(epoch, d_checkpoint_path))
              utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, d_oldest_checkpoint_path, False)
              os.rename(d_oldest_checkpoint_path, d_checkpoint_path)
          else:
            if num_checkpoints > num_checkpoints_to_keep:
              # https://stackoverflow.com/questions/53028607/how-to-remove-the-file-from-trash-in-drive-in-colab
              open(g_oldest_checkpoint_path, 'w').close() # Overwrite and make the file blank
              open(d_oldest_checkpoint_path, 'w').close()
              os.remove(g_oldest_checkpoint_path) # Delete the blank file from google drive will move the file to bin instead
              os.remove(d_oldest_checkpoint_path)      
    global_step += 1
  
  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))

 
def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    with torch.no_grad():
      for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(eval_loader):
        x, x_lengths = x.cuda(0), x_lengths.cuda(0)
        spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
        y, y_lengths = y.cuda(0), y_lengths.cuda(0)

        # remove else
        x = x[:1]
        x_lengths = x_lengths[:1]
        spec = spec[:1]
        spec_lengths = spec_lengths[:1]
        y = y[:1]
        y_lengths = y_lengths[:1]
        break
      y_hat, attn, mask, *_ = generator.module.infer(x, x_lengths, max_len=1000)
      y_hat_lengths = mask.sum([1,2]).long() * hps.data.hop_length

      mel = spec_to_mel_torch(
        spec, 
        hps.data.filter_length, 
        hps.data.n_mel_channels, 
        hps.data.sampling_rate,
        hps.data.mel_fmin, 
        hps.data.mel_fmax)
      y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
      )
    image_dict = {
      "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }
    audio_dict = {
      "gen/audio": y_hat[0,:,:y_hat_lengths[0]]
    }
    if global_step == 0:
      image_dict.update({"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
      audio_dict.update({"gt/audio": y[0,:,:y_lengths[0]]})

    utils.summarize(
      writer=writer_eval,
      global_step=global_step, 
      images=image_dict,
      audios=audio_dict,
      audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()

                           
if __name__ == "__main__":
  main()
