import torch
from dataset import ContextualizedDecompressionLoader
from option import parse_args
from tqdm import tqdm
from trainer import Trainer
from utility import Checkpoint


def main():
    """
    Main entry point for training and validating the RNAN model.
    Handles argument parsing, data loading, training loop, and checkpointing.
    """
    # Parse command-line arguments
    args = parse_args()
    # Initialize checkpointing utility
    checkpoint = Checkpoint(args)
    # Enable cuDNN benchmark for performance if CUDA is available
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    # Prepare the data loader for contextualized decompression
    loader = ContextualizedDecompressionLoader(
        args.dataset, batch_size=args.batch_size, path="../../prepared_data/"
    )
    # Initialize the trainer with arguments, data loader, and checkpoint
    t = Trainer(args, loader, checkpoint)

    try:
        # Progress bar for epochs
        tt = tqdm(desc="Epoch", dynamic_ncols=True, total=args.epochs - t.current_epoch)
        while not t.terminate():
            # Training step
            t.step(is_train=True)
            # Validation step
            is_best, val_loss = t.step(is_train=False)
            # Save checkpoint
            checkpoint.save(t, is_best, args.keep, t.current_epoch)
            tt.update(1)
            tt.set_description("Loss: {}".format(val_loss))
        tt.close()
    except KeyboardInterrupt:
        # Graceful exit on keyboard interrupt
        pass

    checkpoint.done()


if __name__ == "__main__":
    main()
