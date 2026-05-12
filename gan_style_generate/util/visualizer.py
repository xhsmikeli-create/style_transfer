import numpy as np
import sys
import ntpath
import time
from . import util
from pathlib import Path
import wandb
import os
import torch.distributed as dist


def save_images(image_dir, visuals, image_path, aspect_ratio=1.0):
    """Save images to the disk.

    Parameters:
        image_dir (str or Path)  -- directory where images will be saved
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
    """
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    name = Path(image_path[0]).stem

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = f"{name}_{label}.png"
        save_path = image_dir / image_name
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)


class Visualizer:
    """This class includes several functions that can display/save images and print/save logging information.

    It uses wandb for optional logging and saves training images to disk.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: Initialize wandb (if enabled)
        Step 3: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.saved = False
        self.use_wandb = opt.use_wandb
        self.current_epoch = 0
        self.img_dir = Path(opt.checkpoints_dir) / opt.name / "images"

        # Initialize wandb if enabled
        if self.use_wandb:
            # Only initialize wandb on main process (rank 0)
            if not dist.is_initialized() or dist.get_rank() == 0:
                self.wandb_project_name = getattr(opt, "wandb_project_name", "CycleGAN")
                self.wandb_run = wandb.init(project=self.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
                self.wandb_run._label(repo="CycleGAN")
            else:
                self.wandb_run = None

        # create a logging file to store training losses
        self.log_name = Path(opt.checkpoints_dir) / opt.name / "loss_log.txt"
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write(f"================ Training Loss ({now}) ================\n")

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def set_dataset_size(self, dataset_size):
        """Set the dataset size for global step calculation"""
        self.dataset_size = dataset_size

    def _calculate_global_step(self, epoch, epoch_iter):
        """Calculate global step from epoch and epoch_iter"""
        # Assuming epoch starts from 1 and epoch_iter is cumulative within epoch
        return (epoch - 1) * self.dataset_size + epoch_iter

    def display_current_results(self, visuals, epoch: int, total_iters: int, save_result=False):
        """Save current results to wandb and image files."""
        # Only display results on main process (rank 0)
        if "LOCAL_RANK" in os.environ and dist.is_initialized() and dist.get_rank() != 0:
            return

        if self.use_wandb:
            ims_dict = {}
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                wandb_image = wandb.Image(image_numpy, caption=f"{label} - Step {total_iters}")
                ims_dict[f"results/{label}"] = wandb_image
            self.wandb_run.log(ims_dict, step=total_iters)

        if save_result or not self.saved:
            self.saved = True
            self.img_dir.mkdir(parents=True, exist_ok=True)
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = self.img_dir / f"epoch{epoch:03d}_{label}.png"
                util.save_image(image_numpy, img_path)

    def plot_current_losses(self, total_iters, losses):
        """Log current losses to wandb

        Parameters:
            total_iters (int)     -- current training iteration during this epoch
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        # Only plot losses on main process (rank 0)
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        if self.use_wandb:
            self.wandb_run.log(losses, step=total_iters)

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        message = f"[Rank {local_rank}] (epoch: {epoch}, iters: {iters}, time: {t_comp:.3f}, data: {t_data:.3f}) "
        for k, v in losses.items():
            message += f", {k}: {v:.3f}"
        message += "\n"
        print(message)  # print the message on ALL ranks with rank info

        # Only save to log file on main process (rank 0)
        if local_rank == 0:
            with open(self.log_name, "a") as log_file:
                log_file.write(f"{message}\n")  # save the message
