import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .dataset import AEDataset, DeepChessDataset
from .autoencoder import AE
from .deepchess import DeepChess
from .utils import *


def train_autoencoder(args):
    """Trains an autoencoder model.

    Args:
        args (argparse.Namespace): The arguments to be used for training.
    """

    # Set up logging.
    if args.verbose:
        get_logging_config()

    # Set up seeds.
    logging.info(f"Training autoencoder")
    if args.seed is not None:
        seed_everything(args.seed)

    # Set up device.
    if args.cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device("cuda")
        logging.info(f"Using {torch.cuda.get_device_name(0)}.")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU.")

    # Read data.
    train_white = read_txt(args.train_files[0])
    train_black = read_txt(args.train_files[1])
    dev_white = read_txt(args.dev_files[0])
    dev_black = read_txt(args.dev_files[1])

    # Set up data loaders.
    train_dataset = AEDataset(train_white + train_black)
    dev_dataset = AEDataset(dev_white + dev_black)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    logging.info(f"Training on {len(train_dataset):,} positions.")
    logging.info(f"Validating on {len(dev_dataset):,} positions.")

    # Initialize model.
    model = AE().to(device)

    logging.info(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize optimizer, scheduler, and criterion.
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    criterion = nn.MSELoss(reduction="sum")

    # Create save path and logs path.
    save_path = os.path.join(args.save_path, args.comment)
    logs = args.logs + ("" if args.comment is None else f"{args.comment}")
    writer = SummaryWriter(log_dir=logs)

    # Initialize best loss and epoch index.
    best_dev_loss = float("inf")
    best_dev_acc = 0
    epoch_idx = 0
    train_loss_idx = 0

    # Create directories to save models and logs.
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(logs, exist_ok=True)

    # Load model if specified.
    if args.load is not None:
        logging.info(f"Loading model from {args.load}.")
        values = load_state_dict(args.load, model, optimizer, scheduler)
        _ = values[0]
        best_dev_loss = values[1]
        _ = values[2]
        best_dev_acc = values[3]
        epoch_idx = values[4]
        train_loss_idx = values[5]
    else:
        logging.info("No model to load. Training from scratch.")

    # Main training loop.
    for epoch in range(epoch_idx+1, epoch_idx+args.epochs+1):

        # Train model.
        model.train()

        # Initialize total train loss and total train accuracy.
        total_train_loss = 0
        total_train_acc = 0

        # Iterate over training data.
        for i, x in (bar := tqdm(enumerate(train_loader), desc=f"epoch {epoch}", ncols=100, total=len(train_loader))):

            # Convert data to float and move to device.
            x = x.float().to(device)

            # Forward pass.
            output = model(x)
            loss = criterion(output, x)

            # Backward pass.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate loss and accuracy.
            total_train_loss += (loss.item() / len(train_loader))
            correct = (x == output.round()).all(dim=1).sum().item() / len(x)
            total_train_acc += (correct / len(train_loader))

            # Log step loss and learning rate.
            if i % args.log_interval == 0:
                writer.add_scalar("step_loss/train", loss.item(), global_step=train_loss_idx)
                writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], global_step=train_loss_idx)

            if i % (args.log_interval * 10) == 0:
                bar.set_postfix(loss=f"{loss.item():.4f}")

            # Increment train loss index.
            train_loss_idx += 1

        # Log epoch loss and epoch accuracy.
        writer.add_scalar("epoch_loss/train", total_train_loss, global_step=epoch_idx)
        writer.add_scalar("epoch_accuracy/train", total_train_acc, global_step=epoch_idx)

        # Update learning rate.
        scheduler.step()

        # Evaluate on dev data.
        model.eval()
        with torch.no_grad():

            # Initialize total dev loss and total dev accuracy.
            total_dev_loss = 0
            total_dev_acc = 0

            # Iterate over dev data.
            for i, x in (bar := tqdm(enumerate(dev_loader), desc=f"epoch {epoch}", ncols=100, total=len(dev_loader))):

                # Convert data to float and move to device.
                x = x.float().to(device)

                # Forward pass.
                output = model(x)
                loss = criterion(output, x)

                # Calculate loss and accuracy.
                total_dev_loss += (loss.item() / len(dev_loader))
                correct = (x == output.round()).all(dim=1).sum().item() / len(x)
                total_dev_acc += (correct / len(dev_loader))

            # Log epoch loss and epoch accuracy.
            writer.add_scalar("epoch_loss/dev", total_dev_loss, global_step=epoch_idx)
            writer.add_scalar("epoch_accuracy/dev", total_dev_acc, global_step=epoch_idx)

        # Create checkpoint name.
        checkpoint_name = f"epoch-{str(epoch).zfill(2)}_train-{total_train_loss:.4f}_dev-{total_dev_loss:.4f}.pth"

        # Logging train and dev loss.
        if total_dev_loss < best_dev_loss:
            logging.info(f"Dev loss improved from {best_dev_loss:.4f} to {total_dev_loss:.4f}.")
            best_dev_loss = total_dev_loss
        else:
            logging.info(f"Dev loss did not improve from {best_dev_loss:.4f}.")

        if total_dev_acc > best_dev_acc:
            logging.info(f"Dev accuracy improved from {best_dev_acc:.4f} to {total_dev_acc:.4f}.")
            best_dev_acc = total_dev_acc
        else:
            logging.info(f"Dev accuracy did not improve from {best_dev_acc:.4f}.")

        # Save model.
        save_state_dict(
            os.path.join(save_path, checkpoint_name), 
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loss=total_train_loss,
                dev_loss=total_dev_loss,
                train_accuracy=total_train_acc,
                dev_accuracy=total_dev_acc,
                epoch_idx=epoch,
                train_loss_idx=train_loss_idx
        )

        # Increment epoch index.
        epoch_idx += 1

    # Flush and close writer.
    writer.flush()
    writer.close()


def train_deepchess(args):
    """Trains a DeepChess model.

    Args:
        args (argparse.Namespace): The arguments to be used for training.
    """

    # Set up logging.
    if args.verbose:
        get_logging_config()

    # Set up seeds.
    logging.info(f"Training DeepChess model.")
    if args.seed is not None:
        seed_everything(args.seed)

    # Set up device.
    if args.cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device("cuda")
        logging.info(f"Using {torch.cuda.get_device_name(0)}.")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU.")

    # Read data.
    train_white = read_txt(args.train_files[0])
    train_black = read_txt(args.train_files[1])
    dev_white = read_txt(args.dev_files[0])
    dev_black = read_txt(args.dev_files[1])

    # Set up data loaders.
    train_dataset = DeepChessDataset(train_white, train_black, args.train_poses_per_epoch)
    dev_dataset = DeepChessDataset(dev_white, dev_black, args.dev_poses_per_epoch)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    logging.info(f"Training on {args.train_poses_per_epoch:,} positions per epoch.")
    logging.info(f"Validating on {args.dev_poses_per_epoch:,} positions per epoch.")

    # Load autoencoder and initialize DeepChess model.
    autoencoder = AE()
    _ = load_state_dict(args.autoencoder, autoencoder)
    deepchess = DeepChess(autoencoder.encoder).to(device)

    logging.info(f"Total model parameters: {sum(p.numel() for p in deepchess.parameters()):,}")

    # Initialize optimizer, scheduler, and criterion.
    optimizer = optim.Adam(deepchess.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    criterion = nn.BCELoss()

    # Create save path and logs path.
    save_path = os.path.join(args.save_path, args.comment)
    logs = args.logs + ("" if args.comment is None else f"{args.comment}")
    writer = SummaryWriter(log_dir=logs)

    # Initialize best loss and epoch index.
    best_train_loss = float("inf")
    best_dev_loss = float("inf")
    best_train_acc = 0
    best_dev_acc = 0
    epoch_idx = 0
    train_loss_idx = 0

    # Create directories to save models and logs.
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(logs, exist_ok=True)

    # Load model if specified.
    if args.load is not None:
        logging.info(f"Loading model from {args.load}.")
        values = load_state_dict(args.load, deepchess, optimizer, scheduler)
        best_train_loss = values[0]
        best_dev_loss = values[1]
        best_train_acc = values[2]
        best_dev_acc = values[3]
        epoch_idx = values[4]
        train_loss_idx = values[5]
    else:
        logging.info("No model to load. Training from scratch.")

    # Main training loop.
    for epoch in range(epoch_idx+1, epoch_idx+args.epochs+1):

        # Train model.
        deepchess.train()

        # Initialize total train loss and total train accuracy.
        total_train_loss = 0
        total_train_acc = 0

        # Iterate over training data.
        for i, x in (bar := tqdm(enumerate(train_loader), desc=f"epoch {epoch}", ncols=100, total=len(train_loader))):

            # Unpack data, convert to float, and move to device.
            white_pos, black_pos, label = x
            white_pos = white_pos.float().to(device)
            black_pos = black_pos.float().to(device)
            label = label.to(device)

            # Forward pass.
            output = deepchess(white_pos, black_pos)
            loss = criterion(output, label)

            # Backward pass.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate loss and accuracy.
            total_train_loss += (loss.item() / len(train_loader))
            predicted = torch.argmax(output, dim=1)
            correct = (predicted == torch.argmax(label, dim=1)).sum().item() / len(predicted)
            total_train_acc += (correct / len(train_loader))

            # Log step loss and learning rate.
            if i % args.log_interval == 0:
                writer.add_scalar("step_loss/train", loss.item(), global_step=train_loss_idx)
                writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], global_step=train_loss_idx)

            if i % (args.log_interval * 10) == 0:
                bar.set_postfix(loss=f"{loss.item():.4f}")

            # Increment train loss index.
            train_loss_idx += 1

        # Log epoch loss and epoch accuracy.
        writer.add_scalar("epoch_loss/train", total_train_loss, global_step=epoch_idx)
        writer.add_scalar("epoch_accuracy/train", total_train_acc, global_step=epoch_idx)

        # Update learning rate.
        scheduler.step()

        # Evaluate on dev data.
        deepchess.eval()

        # Initialize total dev loss and total dev accuracy.
        total_dev_loss = 0
        total_dev_acc = 0

        # Iterate over dev data.
        with torch.no_grad():
            for i, x in (bar := tqdm(enumerate(dev_loader), desc=f"epoch {epoch}", ncols=100, total=len(dev_loader))):

                # Unpack data, convert to float, and move to device.
                white_pos, black_pos, label = x
                white_pos = white_pos.float().to(device)
                black_pos = black_pos.float().to(device)
                label = label.to(device)

                # Forward pass.
                output = deepchess(white_pos, black_pos)
                loss = criterion(output, label)

                # Calculate loss and accuracy.
                total_dev_loss += (loss.item() / len(dev_loader))
                predicted = torch.argmax(output, dim=1)
                correct = (predicted == torch.argmax(label, dim=1)).sum().item() / len(predicted)
                total_dev_acc += (correct / len(dev_loader))

            # Log epoch loss and epoch accuracy.
            writer.add_scalar("epoch_loss/dev", total_dev_loss, global_step=epoch_idx)
            writer.add_scalar("epoch_accuracy/dev", total_dev_acc, global_step=epoch_idx)
        
        # Logging train and dev loss.
        if total_train_loss < best_train_loss:
            logging.info(f"Train loss improved from {best_train_loss:.4f} to {total_train_loss:.4f}.")
            best_train_loss = total_train_loss
        else:
            logging.info(f"Train loss did not improve from {best_train_loss:.4f}.")

        if total_dev_loss < best_dev_loss:
            logging.info(f"Dev loss improved from {best_dev_loss:.4f} to {total_dev_loss:.4f}.")
            best_dev_loss = total_dev_loss
        else:
            logging.info(f"Dev loss did not improve from {best_dev_loss:.4f}.")

        # Logging train and dev accuracy.
        if total_train_acc > best_train_acc:
            logging.info(f"Train accuracy improved from {best_train_acc:.4f} to {total_train_acc:.4f}.")
            best_train_acc = total_train_acc
        else:
            logging.info(f"Train accuracy did not improve from {best_train_acc:.4f}.")

        if total_dev_acc > best_dev_acc:
            logging.info(f"Dev accuracy improved from {best_dev_acc:.4f} to {total_dev_acc:.4f}.")
            best_dev_acc = total_dev_acc
        else:
            logging.info(f"Dev accuracy did not improve from {best_dev_acc:.4f}.")

        # Create checkpoint name.
        checkpoint_name = f"epoch-{str(epoch).zfill(2)}_loss-{total_train_loss:.4f}_train_acc-{total_train_acc:.4f}_dev_acc-{total_dev_acc:.4f}.pth"
        
        # Save model.
        save_state_dict(
            os.path.join(save_path, checkpoint_name), 
                model=deepchess, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                train_loss=total_train_loss,
                dev_loss=total_dev_loss,
                train_accuracy=total_train_acc,
                dev_accuracy=total_dev_acc,
                epoch_idx=epoch, 
                train_loss_idx=train_loss_idx
        )

        # Increment epoch index.
        epoch_idx += 1

    # Flush and close writer.
    writer.flush()
    writer.close()


def parse_args():
    """Parses command line arguments."""

    parser = argparse.ArgumentParser(description="Train DeepChess models.")

    # Autoencoder arguments.
    parser.add_argument("--train_autoencoder", action="store_true", default=False, help="Train autoencoder.")

    # DeepChess arguments.
    parser.add_argument("--train_deepchess", action="store_true", default=False, help="Train DeepChess model.")
    parser.add_argument("--autoencoder", type=str, help="Path to trained autoencoder model.")
    parser.add_argument("--train_poses_per_epoch", type=int, default=1000000, help="Number of train positions per epoch.")
    parser.add_argument("--dev_poses_per_epoch", type=int, default=100000, help="Number of dev positions per epoch.")

    # Mutual arguments.
    parser.add_argument("--train_files", type=str, nargs="+", help="Path to train files containing positions.")
    parser.add_argument("--dev_files", type=str, nargs="+", help="Path to dev files containing positions.")
    parser.add_argument("--load", type=str, default=None, help="Path to load model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers.")
    parser.add_argument("--gamma", type=float, default=0.9, help="Gamma.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--logs", type=str, default="logs", help="Path to save logs.")
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval.")
    parser.add_argument("--save_path", type=str, default="models", help="Path to save models.")
    parser.add_argument("--comment", type=str, default=None, help="Comment for TensorBoard.")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA.")
    parser.add_argument("--verbose", action="store_true", help="Whether to print logs.")
    return parser.parse_args()


def main():

    # Parse command line arguments.
    args = parse_args()

    # Train specified model.
    if args.train_autoencoder:
        train_autoencoder(args)
    elif args.train_deepchess:
        train_deepchess(args)
    else:
        raise ValueError("Must specify --train_autoencoder or --train_deepchess.")


if __name__ == "__main__":
    main()