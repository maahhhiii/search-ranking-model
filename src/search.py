from src.ranker import SearchRanker


def main():
    ranker = SearchRanker()

    print("Search Ranking Model")
    print("Type 'exit' to stop.\n")

    while True:
        query = input("Search: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        results = ranker.search(query, top_k=5)

        print("\nResults")
        print("-------")
        for i, result in enumerate(results, start=1):
            print(f"{i}. {result['title']}")
            print(f"   Score: {result['score']}")
            print(f"   {result['content']}\n")


if __name__ == "__main__":
    main()
