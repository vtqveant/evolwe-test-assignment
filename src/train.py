import argparse
from datetime import datetime

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

from dataset import HelloEvolweDataset
from src.label_tracker import DictLabelTracker

NUM_LABELS = 71


def train(args, model, tokenizer, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, sample in enumerate(train_loader):
        optimizer.zero_grad()

        labels = sample['intent_idx'].unsqueeze(0).to(device)

        texts = sample['text']
        encoded_input = tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=texts,
            add_special_tokens=True,
            padding='max_length',
            max_length=512,
            return_attention_mask=True,
            return_tensors='pt'
        ).to(device)

        outputs = model(**encoded_input, labels=labels)
        loss, logits = outputs[:2]

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.12f}'.format(
                epoch, batch_idx * len(texts), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

        if batch_idx > 0 and batch_idx % args.snapshot_interval == 0:
            if args.dry_run:
                break
            print('Saving snapshot for epoch {}\n'.format(epoch))
            torch.save(model.state_dict(), '../snapshots/' + datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + '.pth')


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Evolwe Test Assignment :: Intent Classification (BERT)')
    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='input batch size for training (default: 20)')
    parser.add_argument('--test-batch-size', type=int, default=20, metavar='N',
                        help='input batch size for testing (default: 20)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"INFO: Using {device} device")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 0,
                       'pin_memory': True,
                       'shuffle': False}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=NUM_LABELS,
        output_attentions=False,
        output_hidden_states=False
    ).to(device)
    print(model)

    # weight_decay here means L2 regularization, s. https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    train_dataset = HelloEvolweDataset(
        filename='../data/hello_nova_intents_0.2.2.yaml',
        label_tracker=DictLabelTracker()
    )
    train_loader = DataLoader(train_dataset, **train_kwargs)

    for epoch in range(1, args.epochs + 1):
        train(args, model, tokenizer, device, train_loader, optimizer, epoch)
        torch.save(model.state_dict(), '../snapshots/' + datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + '.pth')
        # test(model, device, test_loader)


if __name__ == '__main__':
    main()
