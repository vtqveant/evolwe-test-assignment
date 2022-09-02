"""
1. Retrieve top-k candidate PL snippets for a single NL entry
2. Construct k (NL, PL_i) pairs and pass them to a CodeBERT-based binary classifier to obtain a ranking score
   (a probability that this pair belongs to the positive class)
3. re-rank top-k candidates using the ranking score
4. compute mean reciprocal rank (MRR) using the re-ranked list
"""
import json

from dataset import HelloEvolweDataset
from opensearch import OpenSearchClient

from sentence_encoder import SentenceEncoder
from intent_classifier_bert import IntentClassifier


def compute_classification_score(docstring: str, code: str) -> float:
    # 1. encode for classification...
    # 2. classify
    # 3. return score
    # TODO
    # classifier = CasCodeClassifier()
    pass


# IMPORTANT: Кандидаты на тесте должны выбираться из candidate codes, их число не совпадает с числом testing queries !!!
def main():
    dataset = HelloEvolweDataset(
        data_directory='../data/dataset',
        # languages=['go', 'java', 'javascript', 'php', 'python', 'ruby'],
        languages=['php'],
        partitions=['test'],
    )

    encoder = SentenceEncoder()
    client = OpenSearchClient()

    for document in dataset:
        print(document.docstring)
        print()

        candidates_for_reranking = []

        # 1. retrieve candidates set
        query_embedding = encoder.encode(document.docstring)
        response = client.search_knn(index='code-bert-knn', partition='test', embedding=query_embedding, k=100)
        if response.status_code != 200:
            print("WARNING: request to OpenSearch failed")
            continue
        response_dict = json.loads(response.text)
        results = response_dict["hits"]["hits"]

        # 2. build (NL, PL) pairs for reranking (NL is the same) and compute the score
        for candidate in results:
            candidate_code = candidate["fields"]["code"][0]
            print(candidate_code)

            score = compute_classification_score(document.docstring, candidate_code)
            candidates_for_reranking.append([candidate, score])

        # 3. rerank the candidates using the score

        # 4. compute MRR

        break


if __name__ == '__main__':
    main()
