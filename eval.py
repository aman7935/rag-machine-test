from main import embedding_function
import chromadb

K = 7

questions = [
    ("How much interest was paid during the statement period?",
     "Interest paid during statement period"),

    ("When was installment number 52 bounced?",
     "Installment Bounced, Cheque S. No.: S116456321/1-52"),
]


def main():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("loan_statement")

    ids = collection.get()["ids"]
    texts = collection.get()["documents"]

    total_relevant = 0
    total_retrieved = 0

    for question, snippet in questions:

        correct_ids = []

        for i in range(len(texts)):
            if snippet.lower() in texts[i].lower():
                correct_ids.append(ids[i])

        vector = embedding_function([question])[0]
        results = collection.query(
            query_embeddings=[vector],
            n_results=K
        )

        retrieved_ids = results["ids"][0]

        relevant = 0

        for chunk_id in retrieved_ids:
            if chunk_id in correct_ids:
                relevant += 1

        total_relevant += relevant
        total_retrieved += K

        print(question)
        print("Relevant:", relevant)
        print("Retrieved:", retrieved_ids)
        print()

    recall = total_relevant / len(questions)
    precision = total_relevant / total_retrieved

    print("Recall@5:", recall)
    print("Precision@5:", precision)


if __name__ == "__main__":
    main()
