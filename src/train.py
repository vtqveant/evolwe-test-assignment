import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from datetime import datetime

from dataset import HelloEvolweDataset
from src.label_tracker import LabelTracker


NUM_LABELS = 67


def train(args, model, tokenizer, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()

        labels = batch['intent_idx'].to(device)

        texts = batch['text']
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

        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        if batch_idx % args['log_interval'] == 0:
            print('Train epoch {} ({:.0f}%):\tloss: {:.12f}'.format(
                epoch, 100. * batch_idx / len(train_loader), loss.item())
            )


def evaluate(model, tokenizer, device, test_loader):
    model.eval()

    validation_accuracy = []
    validation_loss = []

    for batch in test_loader:
        labels = batch['intent_idx'].to(device)

        texts = batch['text']
        encoded_input = tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=texts,
            add_special_tokens=True,
            padding='max_length',
            max_length=512,
            return_attention_mask=True,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded_input)
            logits = outputs['logits']

        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        validation_loss.append(loss.item())

        predictions = torch.argmax(logits, dim=1).flatten()
        accuracy = torch.eq(predictions, labels).cpu().numpy().mean()
        validation_accuracy.append(accuracy)

    return np.mean(validation_loss), np.mean(validation_accuracy)


def main():

    # torch.cuda.empty_cache()

    # training settings
    args = {
        'batch_size': 10,
        'epochs': 20,
        'lr': 1e-5,
        'log_interval': 10,
        'snapshot_interval': 100
    }

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"INFO: Using {device} device")

    train_kwargs = {'batch_size': args['batch_size'], 'shuffle': False}
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

    # freeze (some) BERT layers to avoid GPU Out-of-Memory error
    for name, param in model.named_parameters():
        if name.startswith("bert.embeddings"):
            param.requires_grad = False
        if name.startswith("bert.encoder.layer") and not \
                (name.startswith("bert.encoder.layer.8") or
                 name.startswith("bert.encoder.layer.9") or
                 name.startswith("bert.encoder.layer.10") or
                 name.startswith("bert.encoder.layer.11")):
            param.requires_grad = False

    # weight_decay here means L2 regularization, s. https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
    # also skip frozen parameters
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args['lr'], eps=1e-8, weight_decay=1e-4)

    dataset = HelloEvolweDataset(
        filename='../data/dataset.csv',
        label_tracker=LabelTracker()
    )

    # splits
    test_split_portion = 0.2
    n_samples = len(dataset)
    indices = list(range(n_samples))
    split_idx = int(np.floor(test_split_portion * n_samples))

    # shuffle
    random_seed = 42
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_indices, test_indices = indices[split_idx:], indices[:split_idx]

    # samplers
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, sampler=train_sampler, **train_kwargs)
    test_loader = DataLoader(dataset, sampler=test_sampler, **train_kwargs)

    # start where we ended last time
    # model.load_state_dict(torch.load('../snapshots/03-09-2022_22:38:04_e149_lr1e-6.pth'))

    for epoch in range(1, args['epochs'] + 1):
        train(args, model, tokenizer, device, train_loader, optimizer, epoch)
        torch.save(model.state_dict(), '../snapshots/' + datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + '.pth')
        validation_loss, validation_accuracy = evaluate(model, tokenizer, device, test_loader)
        print("Eval. epoch {}:\tloss = {:.12f}, accuracy = {:.4f}".format(epoch, validation_loss, validation_accuracy))


if __name__ == '__main__':
    main()
