import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from datetime import datetime

from dataset import HelloEvolweDataset
from src.label_tracker import LabelTracker


NUM_LABELS = 67


def train(args, model, tokenizer, device, train_loader, optimizer, scheduler, epoch, class_weights):
    model.train()
    class_weights = torch.tensor(class_weights).to(device)

    for batch_idx, sample in enumerate(train_loader):
        optimizer.zero_grad()

        labels = sample['intent_idx'].to(device)

        texts = sample['text']
        encoded_input = tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=texts,
            add_special_tokens=True,
            padding='max_length',
            max_length=512,
            return_attention_mask=True,
            return_tensors='pt'
        ).to(device)

        outputs = model(**encoded_input)
        logits = outputs['logits']

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()
        scheduler.step()

        if batch_idx % args['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.12f}'.format(
                epoch, batch_idx * len(texts), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args['dry_run']:
                break


def main():
    # training settings
    args = {
        'batch_size': 15,
        'epochs': 20,
        'lr': 5e-5,
        'log_interval': 10,
        'dry_run': False,
        'snapshot_interval': 100
    }

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"INFO: Using {device} device")

    train_kwargs = {'batch_size': args['batch_size'], 'shuffle': True}
    if use_cuda:
        train_kwargs.update({'num_workers': 0, 'pin_memory': True})


    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=NUM_LABELS,
        output_attentions=False,
        output_hidden_states=False
    ).to(device)
    # print(model)

    # weight_decay here means L2 regularization, s. https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
    optimizer = AdamW(model.parameters(), lr=args['lr'], eps=1e-8, weight_decay=1e-4)

    train_dataset = HelloEvolweDataset(
        filename='data/train.csv',
        label_tracker=LabelTracker()
    )
    train_loader = DataLoader(train_dataset, **train_kwargs)
    class_weights = train_dataset.get_class_weights()

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=args['epochs'])

    # start where we ended last time
    # model.load_state_dict(torch.load('/content/snapshots/02-09-2022_19:01:31.pth'))

    for epoch in range(1, args['epochs'] + 1):
        train(args, model, tokenizer, device, train_loader, optimizer, scheduler, epoch, class_weights)
        torch.save(model.state_dict(), 'snapshots/' + datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + '.pth')
        # test(model, device, test_loader)


if __name__ == '__main__':
    main()
