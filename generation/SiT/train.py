# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import torch

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from torchvision.transforms import InterpolationMode
from models import SiT_models
from download import find_model
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from train_utils import parse_transport_args
import wandb_utils
from dataset import build_dataset
from torch.cuda.amp import autocast  # (ADDED) for optional bf16 autocast
from torchvision.transforms import functional as TF


import warnings
warnings.simplefilter("ignore", UserWarning)


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):

    def strip_prefix(n: str) -> str:
        if n.startswith("_orig_mod."):
            return n[len("_orig_mod."):]
        return n

    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict((strip_prefix(n), p) for n, p in model.named_parameters())

    for name, param in model_params.items():
        if name in ema_params:
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new SiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."

    rank = dist.get_rank()
    # 优先使用 torchrun/slurm 提供的 LOCAL_RANK；若没有则回退到按卡数取模
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    device = torch.device("cuda", local_rank)

    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(local_rank)

    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    local_batch_size = int(args.global_batch_size // dist.get_world_size())

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., SiT-XL/2 --> SiT-XL-2 (for naming folders)

        # if args.tips is not None:
        #     experiment_name = f"{experiment_index:03d}-{args.tips}-{model_string_name}-" \
        #                       f"{args.path_type}-{args.prediction}-{args.loss_weight}"
        # else:
        
        experiment_name = f"{experiment_index:03d}-{model_string_name}-" \
                            f"{args.path_type}-{args.prediction}-{args.loss_weight}"

        experiment_dir = f"{args.results_dir}/{experiment_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        entity = os.environ["ENTITY"]
        project = os.environ["PROJECT"]
        if args.wandb:
            wandb_utils.initialize(args, entity, experiment_name, project)
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8

    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        class_dropout_prob = args.class_dropout_rate
    )

    if args.num_classes == 1:
        args.cfg_scale = 1.0

    # Note that parameter initialization is done within the SiT constructor
    # ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    ema = deepcopy(model).to(device, memory_format=torch.channels_last)
    model = model.to(device, memory_format=torch.channels_last)


    state_dict = {}
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        state_dict = ckpt

        def _strip(k: str) -> str:
            k = k.replace("module.", "").replace("_orig_mod.", "")
            if k.startswith("model."):
                k = k[len("model."):]
            return k

        if "model" in ckpt:

            model.load_state_dict(ckpt["model"])
            if "ema" in ckpt:
                ema.load_state_dict(ckpt["ema"])
            else:
                ema.load_state_dict(deepcopy(model.state_dict()))

            # args_from_ckpt = ckpt.get("args", None)
        else:

            raw_sd = ckpt.get("state_dict", ckpt)
            cleaned = { _strip(k): v for k, v in raw_sd.items() }
            missing, unexpected = model.load_state_dict(cleaned, strict=False)

            ema.load_state_dict(model.state_dict())
            if dist.get_rank() == 0:
                print(f"[ckpt] loaded bare state_dict (missing={len(missing)}, unexpected={len(unexpected)})")

    requires_grad(ema, False)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )  # default: velocity;
    transport_sampler = Sampler(transport)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device, memory_format=torch.channels_last)
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):

    try:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0, fused=True)
    except TypeError:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0, foreach=True)

    if args.ckpt and isinstance(state_dict, dict) and "opt" in state_dict:
        opt.load_state_dict(state_dict["opt"])

    # Setup data:
    transform = transforms.Compose([
        # transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.Resize(args.image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    # dataset = ImageFolder(args.data_path, transform=transform)
    dataset = build_dataset(transform, args)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )

    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,  # 8~16/进程更稳
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4,  # 2→4 往往更好
    )

    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    use_cfg = (args.cfg_scale > 1.0) and (args.num_classes > 1)

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:

            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)

            with torch.no_grad():
                if args.amp_bf16:  # (ADDED) optional bf16 autocast
                    with autocast(dtype=torch.bfloat16):
                        x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                else:
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)

            model_kwargs = dict(y=y)

            if args.amp_bf16:  # (ADDED) optional bf16 autocast for forward+loss
                with autocast(dtype=torch.bfloat16):
                    loss_dict = transport.training_losses(model, x, model_kwargs)
                    loss = loss_dict["loss"].mean()
            else:
                loss_dict = transport.training_losses(model, x, model_kwargs)
                loss = loss_dict["loss"].mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if args.wandb:
                    wandb_utils.log(
                        {"train loss": avg_loss, "train steps/sec": steps_per_sec},
                        step=train_steps
                    )
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save SiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

            if train_steps % args.sample_every == 0 and train_steps > 0:
                if rank == 0:
                    logger.info("Generating EMA samples...")
                    sample_fn = transport_sampler.sample_ode()

                    n = 36
                    z = torch.randn(n, 4, latent_size, latent_size, device=device)

                    if (args.cfg_scale > 1.0) and (args.num_classes > 1):
                        # 有类且启用 CFG 的情况（FFHQ 单类不会走这里）
                        z_in = torch.cat([z, z], 0)
                        num_classes = model.module.y_embedder.num_classes
                        null_id = num_classes
                        ys = torch.randint(num_classes, size=(n,), device=device)
                        ys = torch.cat([ys, torch.full((n,), null_id, device=device)], 0)
                        model_fn = ema.forward_with_cfg
                        model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale)
                    elif args.num_classes == 1:

                        z_in = z
                        ys = torch.zeros(n, dtype=torch.long, device=device)
                        model_fn = ema.forward
                        model_kwargs = dict(y=ys)
                    else:

                        z_in = z
                        ys = torch.randint(args.num_classes, size=(n,), device=device)
                        model_fn = ema.forward
                        model_kwargs = dict(y=ys)

                    with torch.no_grad():
                        if args.amp_bf16:
                            with autocast(dtype=torch.bfloat16):
                                latents = sample_fn(z_in, model_fn, **model_kwargs)[-1]
                        else:
                            latents = sample_fn(z_in, model_fn, **model_kwargs)[-1]

                        if (args.cfg_scale > 1.0) and (args.num_classes > 1):
                            latents, _ = latents.chunk(2, dim=0)

                        vae.eval()
                        decode_bs = min(8, n)
                        imgs_cpu = []
                        for i in range(0, latents.size(0), decode_bs):
                            zc = latents[i:i + decode_bs]
                            if args.amp_bf16:
                                with autocast(dtype=torch.bfloat16):
                                    dec = vae.decode(zc / 0.18215).sample
                                dec = dec.float()
                            else:
                                dec = vae.decode(zc / 0.18215).sample
                            dec = (dec.clamp(-1, 1) + 1) * 0.5  # [0,1]
                            imgs_cpu.append(dec.to("cpu", non_blocking=True))
                            del dec, zc
                            torch.cuda.empty_cache()

                        samples_cpu = torch.cat(imgs_cpu, dim=0)
                        del imgs_cpu, latents, z, z_in, ys
                        torch.cuda.empty_cache()

                    if args.wandb:

                        wandb_utils.log_image(samples_cpu, train_steps)

                    logger.info("Generating EMA samples done.")

                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train SiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256,512,1024], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--sample-every", type=int, default=10_000)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a custom SiT checkpoint")
    parser.add_argument("--amp-bf16", action="store_true",  # (ADDED) enable bf16 autocast when set
                        help="Enable bf16 autocast for faster/leaner training on supported GPUs")

    parser.add_argument("--class_dropout_rate", type=int, default = 0)

    parser.add_argument("--tips", type=str, default=None)

    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)
