"""
Read examples from dataset, encode with BERT and store to index
"""
import torch.cuda
from transformers import BertTokenizer, BertModel


class SentenceEncoder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"INFO: Using {self.device} device")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.model.to(self.device)

    def encode(self, text):
        encoded_input = self.tokenizer.encode_plus(
            text=text[:512],
            add_special_tokens=True,
            padding='max_length',
            max_length=512,
            return_attention_mask=True,
            return_tensors='pt'
        ).to(self.device)

        input_ids = encoded_input['input_ids']
        print(input_ids)
        attention_mask = encoded_input['attention_mask']
        print(attention_mask)

        output = self.model(**encoded_input)
        hidden_states = output[1]

        cls_embedding = hidden_states[0]
        return cls_embedding.squeeze().tolist()


def main():
    # dataset = CodeSearchNetDataset(
    #     data_directory='../data/dataset',
    #     languages=['go', 'java', 'javascript', 'php', 'python', 'ruby'],
    #     partitions=['train', 'valid', 'test'],
    # )

    encoder = SentenceEncoder()
    embedding = encoder.encode("Who are you?")
    print(embedding)


if __name__ == '__main__':
    main()
