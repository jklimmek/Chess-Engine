import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .dataset import DBNDataset, DeepChessDataset
from .pos2vec import Pos2VecLayer
from .deepchess import DeepChess
from .utils import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


# TODO: Finish training Pos2Vec.
# TODO: Implement sampling positions from DeepChessDataset.
# TODO: Finish Implementing DeepChess training.
# TODO: Add comments for all functions.


def train_dbn_layer(args):
    if args.verbose:
        get_logging_config()

    logging.info(f"Training single layer of DBN: {args.layer[0]} -> {args.layer[1]}")
    seed_everything(args.seed)

    if args.cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device("cuda")
        logging.info(f"Using {torch.cuda.get_device_name(0)}.")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU.")

    train_white = read_txt(args.train_files[0])
    train_black = read_txt(args.train_files[1])
    dev_white = read_txt(args.dev_files[0])
    dev_black = read_txt(args.dev_files[1])
    train_dataset = DBNDataset(train_white + train_black, convert_mode=args.convert_mode)
    dev_dataset = DBNDataset(dev_white + dev_black, convert_mode=args.convert_mode)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    logging.info(f"Training on {len(train_dataset):,} positions.")
    logging.info(f"Validating on {len(dev_dataset):,} positions.")

    model = Pos2VecLayer(args.layer[0], args.layer[1]).to(device)

    extractors = []
    if args.previous_layers is not None and args.previous_models is not None:
        sizes = [(args.previous_layers[i], args.previous_layers[i+1]) for i in range(len(args.previous_layers)-1)]

        for i, (size, model_path) in enumerate(zip(sizes, args.previous_models)):
            extractor = Pos2VecLayer(size[0], size[1]).to(device)
            _ = load_state_dict(model_path, extractor)
            extractor.eval()
            extractors.append(extractor)
    
    logging.info(f"Loaded {len(extractors)} extractor(s).")

    logging.info(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    criterion = nn.MSELoss(reduction="sum")

    save_path = os.path.join(args.save_path, f"{args.layer[0]}_to_{args.layer[1]}")
    logs = os.path.join(args.logs, f"{args.layer[0]}_to_{args.layer[1]}")
    logs = logs + ("" if args.comment is None else f"_{args.comment}")
    writer = SummaryWriter(log_dir=logs)

    best_loss = float("inf")
    epoch_idx = 0
    train_loss_idx = 0

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(logs, exist_ok=True)

    if args.load is not None:
        logging.info(f"Loading model from {args.load}.")
        _, best_loss, epoch_idx, train_loss_idx = load_state_dict(args.load, model, optimizer, scheduler)
    else:
        logging.info("No model to load. Training from scratch.")

    for epoch in range(epoch_idx+1, epoch_idx+args.epochs+1):
        model.train()
        total_train_loss = 0
        for i, x in (bar := tqdm(enumerate(train_loader), desc=f"epoch {epoch}", ncols=100, total=len(train_loader))):
            x = x.float().to(device)

            if len(extractors) > 0:
                with torch.no_grad():
                    for extractor in extractors:
                        x = extractor.encode(x)

            output = model(x)
            loss = criterion(output, x)
            total_train_loss += (loss.item() / len(train_loader))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % args.log_interval == 0:
                writer.add_scalar("step_loss/train", loss.item(), global_step=train_loss_idx)
                writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], global_step=train_loss_idx)
            if i % (args.log_interval * 10) == 0:
                bar.set_postfix(loss=f"{loss.item():.4f}")
            train_loss_idx += 1
        writer.add_scalar("epoch_loss/train", total_train_loss, global_step=epoch_idx)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            total_dev_loss = 0
            for i, x in (bar := tqdm(enumerate(dev_loader), desc=f"epoch {epoch}", ncols=100, total=len(dev_loader))):
                x = x.float().to(device)

                if len(extractors) > 0:
                    for extractor in extractors:
                        x = extractor.encode(x)

                output = model(x)
                loss = criterion(output, x)
                total_dev_loss += (loss.item() / len(dev_loader))
            writer.add_scalar("epoch_loss/dev", total_dev_loss, global_step=epoch_idx)

        checkpoint_name = f"epoch-{str(epoch).zfill(2)}_train-{total_train_loss:.4f}_dev-{total_dev_loss:.4f}.pth"
        if total_dev_loss < best_loss:
            logging.info(f"Dev loss improved from {best_loss:.4f} to {total_dev_loss:.4f}.")
            best_loss = total_dev_loss
        else:
            logging.info(f"Dev loss did not improve from {best_loss:.4f}.")
        save_state_dict(
            os.path.join(save_path, checkpoint_name), 
                model, 
                optimizer, 
                scheduler, 
                total_train_loss,
                total_dev_loss, 
                epoch, 
                train_loss_idx
        )

        epoch_idx += 1

    writer.flush()
    writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepChess models.")

    # Deep Belief Network arguments.
    parser.add_argument("--dbn", action="store_true", default=False, help="Train autoencoder.")
    parser.add_argument("--layer", type=int, nargs="+", help="Input and output layer sizes of DBN.")
    parser.add_argument("--convert_mode", type=str, choices=["none", "fen_to_array"], default="none", help="Convert mode for DBN dataset.")
    parser.add_argument("--previous_layers", type=int, default=None, nargs="+", help="Sizes of layers of previous models.")
    parser.add_argument("--previous_models", type=str, default=None, nargs="+", help="Path to previous models to load.")

    # DeepChess arguments.
    parser.add_argument("--deepchess", action="store_true", default=False, help="Train DeepChess model.")
    parser.add_argument("--pos2vec", type=str, help="Path to trained pos2vec model.")

    # Mutual arguments.
    parser.add_argument("--train_files", type=str, nargs="+", help="Path to train files containing positions.")
    parser.add_argument("--dev_files", type=str, nargs="+", help="Path to dev files containing positions.")
    parser.add_argument("--load", type=str, default=None, help="Path to load model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers.")
    parser.add_argument("--gamma", type=float, default=0.9, help="Gamma.")
    parser.add_argument("--seed", type=int, default=201, help="Random seed.")
    parser.add_argument("--logs", type=str, default="logs", help="Path to save logs.")
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval.")
    parser.add_argument("--save_path", type=str, default="models", help="Path to save models.")
    parser.add_argument("--save_mode", type=str, choices=["best", "all"], default="best", help="Save mode.")
    parser.add_argument("--comment", type=str, default=None, help="Comment for TensorBoard.")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA.")
    parser.add_argument("--verbose", action="store_true", help="Verbose.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dbn:
        train_dbn_layer(args)
    # elif args.deepchess:
    #     train_deepchess(args)
    else:
        raise ValueError("Must specify --autoencoder or --deepchess.")


if __name__ == "__main__":
    main()