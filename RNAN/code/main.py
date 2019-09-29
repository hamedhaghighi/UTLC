import torch
from tqdm import tqdm

from trainer import Trainer
from option import parse_args
from utility import Checkpoint
from dataset import ContextualizedDecompressionLoader


def main():
    args = parse_args()
    checkpoint = Checkpoint(args)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    loader = ContextualizedDecompressionLoader(args.dataset, batch_size=args.batch_size, path='../../prepared_data/')
    t = Trainer(args, loader, checkpoint)

    try:
        tt = tqdm(desc='Epoch', dynamic_ncols=True, total=args.epochs - t.current_epoch)
        while not t.terminate():
            t.step(is_train=True)
            is_best, val_loss = t.step(is_train=False)
            checkpoint.save(t, is_best, args.keep, t.current_epoch)
            tt.update(1)
            tt.set_description('Loss: {}'.format(val_loss))
        tt.close()
    except KeyboardInterrupt:
        pass

    checkpoint.done()


if __name__ == '__main__':
    main()
