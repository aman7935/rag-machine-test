# a quick way to check if the retriever is working.
# for each question i also write a small part of the answer text.
# if that text is not in the top results, something is wrong.

from main import embedding_function
import chromadb

K = 5

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

    found_count = 0

    for question, snippet in questions:
        # first, find which chunks have the answer in them
        correct_ids = []
        for i in range(len(ids)):
            if snippet.lower() in texts[i].lower():
                correct_ids.append(ids[i])

        if len(correct_ids) == 0:
            print("! could not find this anywhere:", snippet)
            continue

        # now embed the question and ask chroma for the closest chunks
        query_vector = embedding_function([question])[0]
        results = collection.query(query_embeddings=[query_vector], n_results=K)

        retrieved_ids = results["ids"][0]

        # check if any of the correct chunks made it into the results
        matched = False
        for chunk_id in retrieved_ids:
            if chunk_id in correct_ids:
                matched = True

        if matched:
            found_count += 1

        print(question)
        print("  top results:", retrieved_ids)
        print("  answer is in:", correct_ids)

        if matched:
            print("  -> yes, the answer was found")
        else:
            print("  -> no, it missed it")

        print()

    print(str(found_count) + "/" + str(len(questions)) + " questions passed")


if __name__ == "__main__":
    main()