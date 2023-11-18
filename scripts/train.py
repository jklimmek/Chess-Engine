import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .dataset import AEDataset, DeepChessDataset
from .pos2vec import Pos2Vec
from .utils import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def train_autoencoder(args):
    if args.verbose:
        get_logging_config()
        logging.info("Training autoencoder.")
    seed_everything(args.seed)

    white_poses = read_txt(args.files[0])
    black_poses = read_txt(args.files[1])

    sample_train = lambda x: list(np.random.choice(x, args.num_train_pos, replace=False))
    sample_dev = lambda x: list(np.random.choice(x, args.num_dev_pos, replace=False))
    train_white_poses = sample_train(white_poses)
    train_black_poses = sample_train(black_poses)
    dev_white_poses = sample_dev(white_poses)
    dev_black_poses = sample_dev(black_poses)

    if args.verbose:
        logging.info(f"Sampled {args.num_train_pos} training positions.")
        logging.info(f"Sampled {args.num_dev_pos} dev positions.")

    ae_train_dataset = AEDataset(train_white_poses + train_black_poses)
    ae_dev_dataset = AEDataset(dev_white_poses + dev_black_poses)
    ae_train_loader = DataLoader(ae_train_dataset, batch_size=args.batch_size, shuffle=True)
    ae_dev_loader = DataLoader(ae_dev_dataset, batch_size=args.batch_size, shuffle=False)

    if args.cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device("cuda")
        if args.verbose:
            logging.info(f"Using {torch.cuda.get_device_name(0)}.")
    else:
        device = torch.device("cpu")
        if args.verbose:
            logging.info("Using CPU.")

    model = Pos2Vec().to(device)
    if args.verbose:
        logging.info(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    criterion = nn.BCELoss()

    writer = SummaryWriter(log_dir=args.logs)

    best_loss = float("inf")
    epoch_idx = 0
    train_loss_idx = 0
    last_name = None

    os.makedirs(args.save_path, exist_ok=True)

    if args.load is not None:
        if args.verbose:
            logging.info(f"Loading model from {args.load}.")
        best_loss, epoch_idx, train_loss_idx = load_state_dict(args.load, model, optimizer, scheduler)
        last_name = os.path.basename(args.load)
    else:
        if args.verbose:
            logging.info("No model to load. Training from scratch.")

    for epoch in range(epoch_idx, epoch_idx+args.epochs):
        model.train()
        total_train_loss = 0
        for i, x in (bar := tqdm(enumerate(ae_train_loader), desc=f"epoch {epoch}", ncols=100, total=len(ae_train_loader))):
            x = x.float().to(device)
            output = model(x)
            loss = criterion(output, x)
            total_train_loss += (loss.item() / len(ae_train_loader))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % args.log_interval == 0:
                writer.add_scalar("step_loss/train", loss.item(), global_step=train_loss_idx)
                writer.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step=train_loss_idx)
                bar.set_postfix(loss=f"{loss.item():.4f}")
            train_loss_idx += 1
        writer.add_scalar("epoch_loss/train", total_train_loss, global_step=epoch_idx)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            total_dev_loss = 0
            for i, x in (bar := tqdm(enumerate(ae_dev_loader), desc=f"epoch {epoch}", ncols=100, total=len(ae_dev_loader))):
                x = x.float().to(device)
                output = model(x)
                loss = criterion(output, x)
                total_dev_loss += (loss.item() / len(ae_dev_loader))
            writer.add_scalar("step_loss/dev", total_dev_loss, global_step=epoch_idx)

        checkpoint_name = f"epoch-{epoch}_train-{total_train_loss:.4f}_dev-{total_dev_loss:.4f}.pth"
        if args.save_mode == "best":
            if total_dev_loss < best_loss:
                if args.verbose:
                    logging.info(f"Dev loss improved from {best_loss:.4f} to {total_dev_loss:.4f}.")
                best_loss = total_dev_loss
                save_state_dict(
                    os.path.join(args.save_path, checkpoint_name), 
                    model, 
                    optimizer, 
                    scheduler, 
                    total_train_loss,
                    best_loss, 
                    epoch, 
                    train_loss_idx
                )
                if last_name is not None:
                    os.remove(os.path.join(args.save_path, last_name))
                last_name = checkpoint_name
            else:
                if args.verbose:
                    logging.info(f"Dev loss did not improve from {best_loss:.4f}.")

        if args.save_mode == "all":
            save_state_dict(
                os.path.join(args.save_path, checkpoint_name), 
                    model, 
                    optimizer, 
                    scheduler, 
                    total_train_loss,
                    total_dev_loss, 
                    epoch, 
                    train_loss_idx
            )

        epoch_idx += 1






def train_deepchess():
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepChess models.")
    parser.add_argument("--autoencoder", action="store_true", default=False, help="Train autoencoder.") #
    parser.add_argument("--deepchess", action="store_true", default=False, help="Train DeepChess model.")
    parser.add_argument("--num_train_pos", type=int, default=1000000, help="Number of train positions for autoencoder.") #
    parser.add_argument("--num_dev_pos", type=int, default=10000, help="Number of dev positions for autoencoder.") #
    parser.add_argument("--files", type=str, nargs="+", help="Path to files containing positions for autoencoder.") #
    parser.add_argument("--load", type=str, default=None, help="Path to load model.") #
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.") #
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.") #
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.") #
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum.") #
    parser.add_argument("--gamma", type=float, default=0.9, help="Gamma.") #
    parser.add_argument("--seed", type=int, default=201, help="Random seed.") #
    parser.add_argument("--logs", type=str, default="logs", help="Path to save logs.") #
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval.") #
    parser.add_argument("--save_path", type=str, default="models", help="Path to save models.") #
    parser.add_argument("--save_mode", type=str, choices=["best", "all"], default="best", help="Save mode.") #
    parser.add_argument("--cuda", action="store_true", help="Use CUDA.") #
    parser.add_argument("--verbose", action="store_true", help="Verbose.") # 
    return parser.parse_args()


def main():
    args = parse_args()
    if args.autoencoder:
        train_autoencoder(args)
    elif args.deepchess:
        train_deepchess(args)
    else:
        raise ValueError("Must specify --autoencoder or --deepchess.")


if __name__ == "__main__":
    main()