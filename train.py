import numpy as np
import torch
# from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import argparse
from tqdm import tqdm

from config import device, print_freq, vocab_size, sos_id, eos_id
from data_gen import AiShellDataset, pad_collate
from transformer.decoder import Decoder
from transformer.encoder import Encoder
from transformer.loss import cal_performance
from transformer.optimizer import TransformerOptimizer
from transformer.transformer import Transformer
from utils import parse_args, save_checkpoint, AverageMeter, get_logger


def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    # writer = SummaryWriter()
    epochs_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        # model
        encoder = Encoder(args.d_input * args.LFR_m, args.n_layers_enc, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout, pe_maxlen=args.pe_maxlen)
        decoder = Decoder(sos_id, eos_id, vocab_size,
                          args.d_word_vec, args.n_layers_dec, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout,
                          tgt_emb_prj_weight_sharing=args.tgt_emb_prj_weight_sharing,
                          pe_maxlen=args.pe_maxlen)
        model = Transformer(encoder, decoder)
        # print(model)
        # model = nn.DataParallel(model)

        # optimizer
        optimizer = TransformerOptimizer(
            torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09))

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    logger = get_logger()

    # Move to GPU, if available
    model = model.to(device)

    # Custom dataloaders
    train_dataset = AiShellDataset(args, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=pad_collate,
                                               pin_memory=True, shuffle=True, num_workers=args.num_workers)
    valid_dataset = AiShellDataset(args, 'dev')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=pad_collate,
                                               pin_memory=True, shuffle=False, num_workers=args.num_workers)

    # Epochs
    for epoch in range(start_epoch, args.epochs):
        # One epoch's training
        train_loss = train(train_loader=train_loader,
                           model=model,
                           optimizer=optimizer,
                           epoch=epoch,
                           logger=logger)
        # writer.add_scalar('model/train_loss', train_loss, epoch)

        lr = optimizer.lr
        print('\nLearning rate: {}'.format(lr))
        # writer.add_scalar('model/learning_rate', lr, epoch)
        step_num = optimizer.step_num
        print('Step num: {}\n'.format(step_num))

        # One epoch's validation
        valid_loss = valid(valid_loader=valid_loader,
                           model=model,
                           logger=logger)
        # writer.add_scalar('model/valid_loss', valid_loss, epoch)

        # Check if there was an improvement
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)


def train(train_loader, model, optimizer, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()

    # Batches
    for i, (data) in enumerate(train_loader):
        # Move to GPU, if available
        padded_input, padded_target, input_lengths = data
        padded_input = padded_input.to(device)
        padded_target = padded_target.to(device)
        input_lengths = input_lengths.to(device)

        # Forward prop.
        pred, gold = model(padded_input, input_lengths, padded_target)
        loss, n_correct = cal_performance(pred, gold, smoothing=args.label_smoothing)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(epoch, i, len(train_loader), loss=losses))

    return losses.avg


def valid(valid_loader, model, logger):
    model.eval()

    losses = AverageMeter()

    # Batches
    for data in tqdm(valid_loader):
        # Move to GPU, if available
        padded_input, padded_target, input_lengths = data
        padded_input = padded_input.to(device)
        padded_target = padded_target.to(device)
        input_lengths = input_lengths.to(device)

        with torch.no_grad():
            # Forward prop.
            pred, gold = model(padded_input, input_lengths, padded_target)
            loss, n_correct = cal_performance(pred, gold, smoothing=args.label_smoothing)

        # Keep track of metrics
        losses.update(loss.item())

    # Print status
    logger.info('\nValidation Loss {loss.val:.5f} ({loss.avg:.5f})\n'.format(loss=losses))

    return losses.avg


def main():
    parser = argparse.ArgumentParser(description='Speech Transformer')
    # Low Frame Rate (stacking and skipping frames)
    parser.add_argument('--LFR_m', default=4, type=int,
                        help='Low Frame Rate: number of frames to stack')
    parser.add_argument('--LFR_n', default=3, type=int,
                        help='Low Frame Rate: number of frames to skip')
    # Network architecture
    # encoder
    # TODO: automatically infer input dim
    parser.add_argument('--d_input', default=80, type=int,
                        help='Dim of encoder input (before LFR)')
    parser.add_argument('--n_layers_enc', default=6, type=int,
                        help='Number of encoder stacks')
    parser.add_argument('--n_head', default=8, type=int,
                        help='Number of Multi Head Attention (MHA)')
    parser.add_argument('--d_k', default=64, type=int,
                        help='Dimension of key')
    parser.add_argument('--d_v', default=64, type=int,
                        help='Dimension of value')
    parser.add_argument('--d_model', default=512, type=int,
                        help='Dimension of model')
    parser.add_argument('--d_inner', default=2048, type=int,
                        help='Dimension of inner')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='Dropout rate')
    parser.add_argument('--pe_maxlen', default=5000, type=int,
                        help='Positional Encoding max len')
    # decoder
    parser.add_argument('--d_word_vec', default=512, type=int,
                        help='Dim of decoder embedding')
    parser.add_argument('--n_layers_dec', default=6, type=int,
                        help='Number of decoder stacks')
    parser.add_argument('--tgt_emb_prj_weight_sharing', default=1, type=int,
                        help='share decoder embedding with decoder projection')
    # Loss
    parser.add_argument('--label_smoothing', default=0.1, type=float,
                        help='label smoothing')

    # Training config
    parser.add_argument('--epochs', default=150, type=int,
                        help='Number of maximum epochs')
    # minibatch
    parser.add_argument('--shuffle', default=1, type=int,
                        help='reshuffle the data at every epoch')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='Batch size')
    parser.add_argument('--batch_frames', default=0, type=int,
                        help='Batch frames. If this is not 0, batch size will make no sense')
    parser.add_argument('--maxlen-in', default=800, type=int, metavar='ML',
                        help='Batch size is reduced if the input sequence length > ML')
    parser.add_argument('--maxlen-out', default=150, type=int, metavar='ML',
                        help='Batch size is reduced if the output sequence length > ML')
    parser.add_argument('--num-workers', default=4, type=int,
                        help='Number of workers to generate minibatch')
    # optimizer
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('--k', default=0.2, type=float,
                        help='tunable scalar multiply to learning rate')
    parser.add_argument('--warmup_steps', default=4000, type=int,
                        help='warmup steps')

    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    args = parser.parse_args()

    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()


