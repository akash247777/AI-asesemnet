import argparse

from src.rag import ingest_pdf_into_qdrant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--collection", default=None, help="Qdrant collection name")
    args = parser.parse_args()

    res = ingest_pdf_into_qdrant(args.pdf, args.collection)
    print(f"Ingested {res['num_chunks']} chunks into collection '{res['collection']}'.")


if __name__ == "__main__":
    main()


