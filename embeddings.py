"""Embed document chunks and questions, then retrieve the closest passages."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypedDict

from chunking import DocumentChunk, chunk_pages
from loader import load_documents

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class RankedChunk(TypedDict):
    source: str
    page: int
    chunk: int
    score: float


def load_embedding_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """Load the sentence-transformers model used for passages and questions."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Embedding requires sentence-transformers; install dependencies with "
            "`pip install -r requirements.txt`."
        ) from error

    return SentenceTransformer(model_name)


def _encode(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """Turn text strings into normalized vectors of numbers."""
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return [[float(value) for value in vector] for vector in vectors]


def embed_passages(
    model: SentenceTransformer,
    passages: list[str],
) -> list[list[float]]:
    """Embed document text without a query instruction prefix."""
    return _encode(model, passages)


def embed_query(model: SentenceTransformer, question: str) -> list[float]:
    """Embed one question using BGE's recommended query instruction."""
    return _encode(model, [QUERY_PREFIX + question])[0]


def rank_chunks(
    chunks: list[DocumentChunk],
    passage_vectors: list[list[float]],
    query_vector: list[float],
    k: int,
) -> list[RankedChunk]:
    """Return the k most similar chunks; normalized-vector dot product is cosine similarity."""
    if k <= 0:
        raise ValueError("k must be greater than zero")
    if len(chunks) != len(passage_vectors):
        raise ValueError("Each chunk must have exactly one embedding")

    ranked: list[RankedChunk] = []
    for chunk, passage_vector in zip(chunks, passage_vectors):
        if len(passage_vector) != len(query_vector):
            raise ValueError("Query and document embeddings must have matching dimensions")
        score = sum(left * right for left, right in zip(passage_vector, query_vector))
        ranked.append(
            {
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk": chunk["chunk"],
                "score": score,
            }
        )
    ranked.sort(key=lambda result: result["score"], reverse=True)
    return ranked[:k]


def plot_embeddings(
    passage_vectors: list[list[float]],
    query_vector: list[float],
    chunks: list[DocumentChunk],
    results: list[RankedChunk],
    dimensions: int = 2,
) -> None:
    """Plot passage and query vectors after reducing them to 2D or 3D with PCA."""
    if dimensions not in {2, 3}:
        raise ValueError("Plot dimensions must be 2 or 3")
    if len(passage_vectors) != len(chunks):
        raise ValueError("Each chunk must have exactly one embedding")
    if len(passage_vectors) < dimensions:
        raise ValueError(f"At least {dimensions} passages are required for this plot")

    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError as error:
        raise RuntimeError(
            "Plotting requires matplotlib and scikit-learn; install dependencies with "
            "`pip install -r requirements.txt`."
        ) from error

    vectors = passage_vectors + [query_vector]
    pca = PCA(n_components=dimensions)
    coordinates = pca.fit_transform(vectors)
    passage_coordinates = coordinates[:-1]
    query_coordinates = coordinates[-1]

    figure = plt.figure()
    if dimensions == 3:
        axis = figure.add_subplot(111, projection="3d")
    else:
        axis = figure.add_subplot(111)

    axis.scatter(
        *passage_coordinates.T,
        color="lightgray",
        alpha=0.6,
        label="Other passages",
    )
    axis.scatter(
        *query_coordinates,
        color="red",
        marker="*",
        s=180,
        label="Question",
    )
    for rank, result in enumerate(results, start=1):
        coordinates_for_chunk = passage_coordinates[result["chunk"]]
        axis.scatter(
            *coordinates_for_chunk,
            s=90,
            label=f"Top {rank} (score {result['score']:.3f})",
        )
        axis.text(
            *coordinates_for_chunk,
            f"Top {rank}: {result['source']} p.{result['page']}",
        )

    component_labels = [
        f"PC{index + 1} ({variance:.1%})"
        for index, variance in enumerate(pca.explained_variance_ratio_)
    ]
    axis.set_xlabel(component_labels[0])
    axis.set_ylabel(component_labels[1])
    if dimensions == 3:
        axis.set_zlabel(component_labels[2])
    axis.set_title(f"Question and passage embeddings projected to {dimensions}D")
    axis.legend()
    plt.tight_layout()
    plt.show()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        default="What were Tradeweb's total revenues in Q2 2026?",
        help="Question text to embed and search for",
    )
    parser.add_argument("--k", type=int, default=3, help="Number of passages to show")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show a PCA plot of the question and passage embeddings",
    )
    parser.add_argument(
        "--plot-dimensions",
        type=int,
        choices=(2, 3),
        default=2,
        help="Dimensions for the optional PCA plot (2 or 3)",
    )
    args = parser.parse_args(argv)
    if args.k <= 0:
        parser.error("--k must be greater than zero")

    pages = load_documents()
    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError("No document chunks found; check the files under data/")

    model = load_embedding_model()
    passage_vectors = embed_passages(model, [chunk["text"] for chunk in chunks])
    query_vector = embed_query(model, args.query)
    results = rank_chunks(chunks, passage_vectors, query_vector, args.k)
    print(f"Question: {args.query}")
    print(f"Embedded {len(chunks)} passages into {len(passage_vectors[0])}-number vectors.")
    print(f"Question vector: {len(query_vector)} numbers")
    print(f"First passage vector values: {[round(value, 4) for value in passage_vectors[0][:8]]}")
    print(f"First 8 vector values: {[round(value, 4) for value in query_vector[:8]]}")
    print("\nClosest passages:")
    for rank, result in enumerate(results, start=1):
        chunk = chunks[result["chunk"]]
        print(
            f"\n{rank}. score={result['score']:.4f} | "
            f"{result['source']} page {result['page']}"
        )
        print(chunk["text"][:500])
    if args.plot:
        plot_embeddings(
            passage_vectors,
            query_vector,
            chunks,
            results,
            dimensions=args.plot_dimensions,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
