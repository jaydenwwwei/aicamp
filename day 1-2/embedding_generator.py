import argparse
import random
import re
import sys


SENTENCE_ENDINGS = {".", "!", "?"}


def load_text(path):
    """Load text from a UTF-8 file."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        raise ValueError(f"Input file not found: {path}")
    except OSError as error:
        raise ValueError(f"Could not read input file: {error}")


def tokenize(text):
    """Split text into words and basic punctuation tokens."""
    return re.findall(r"\w+(?:['-]\w+)*|[.!?]", text)


def join_tokens(tokens):
    """Join tokens while keeping punctuation attached to the previous word."""
    sentence = ""
    for token in tokens:
        if token in SENTENCE_ENDINGS:
            sentence = sentence.rstrip() + token + " "
        else:
            sentence += token + " "
    return sentence.strip()


def make_blocks(tokens, block_size, next_size):
    """Create overlapping context blocks and the continuation tokens after each block."""
    blocks = []
    continuations = []

    if len(tokens) <= block_size:
        return blocks, continuations

    for index in range(len(tokens) - block_size):
        block = tokens[index:index + block_size]
        continuation = tokens[index + block_size:index + block_size + next_size]
        if continuation:
            blocks.append(block)
            continuations.append(continuation)

    return blocks, continuations


def load_embedding_model(model_name):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Missing dependency: sentence-transformers. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from error

    return SentenceTransformer(model_name)


def embed_blocks(model, blocks):
    texts = [join_tokens(block) for block in blocks]
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def normalize_embeddings(embeddings):
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "Missing dependency: numpy. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from error

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def cosine_similarities(query_embedding, normalized_embeddings):
    query_norm = query_embedding / max(float((query_embedding ** 2).sum() ** 0.5), 1e-12)
    return normalized_embeddings @ query_norm


def choose_similar_block(query_embedding, normalized_embeddings, top_k, rng):
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "Missing dependency: numpy. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from error

    similarities = cosine_similarities(query_embedding, normalized_embeddings)
    candidate_count = min(top_k, len(similarities))
    if candidate_count < 1:
        return None

    candidate_indexes = np.argpartition(similarities, -candidate_count)[-candidate_count:]
    candidate_indexes = sorted(
        candidate_indexes,
        key=lambda index: similarities[index],
        reverse=True,
    )
    return int(rng.choice(candidate_indexes))


def generate_sentence(
    model,
    blocks,
    continuations,
    normalized_embeddings,
    block_size,
    max_words,
    top_k,
    rng,
):
    """Generate one sentence-like output using embedding similarity retrieval."""
    if not blocks:
        return ""

    start_index = rng.randrange(len(blocks))
    output = list(blocks[start_index])

    while len(output) < max_words:
        context = output[-block_size:]
        context_text = join_tokens(context)
        query_embedding = model.encode(context_text, convert_to_numpy=True, show_progress_bar=False)
        match_index = choose_similar_block(query_embedding, normalized_embeddings, top_k, rng)
        if match_index is None:
            break

        added_any = False
        for token in continuations[match_index]:
            if len(output) >= max_words:
                break
            output.append(token)
            added_any = True
            if token in SENTENCE_ENDINGS:
                return join_tokens(output)

        if not added_any:
            break

    if output and output[-1] not in SENTENCE_ENDINGS:
        output.append(".")
    return join_tokens(output)


def project_embeddings(embeddings, seed):
    try:
        from sklearn.decomposition import PCA
    except ImportError as error:
        raise RuntimeError(
            "Missing dependency: scikit-learn. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from error

    if len(embeddings) < 2:
        raise ValueError("Need at least 2 blocks to visualize embeddings.")

    pca = PCA(n_components=2, random_state=seed)
    return pca.fit_transform(embeddings)


def shorten_label(tokens, max_length=42):
    text = join_tokens(tokens)
    if len(text) <= max_length:
        return text
    return text[:max_length - 3].rstrip() + "..."


def plot_embeddings(points, blocks, model_name, block_size, label_points, max_labels):
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Missing dependency: matplotlib. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from error

    x_values = points[:, 0]
    y_values = points[:, 1]

    plt.figure(figsize=(10, 7))
    plt.scatter(x_values, y_values, alpha=0.75)
    plt.title(f"Embedding Blocks - {model_name} - block size {block_size} - PCA")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True, alpha=0.25)

    if label_points:
        for index, block in enumerate(blocks[:max_labels]):
            plt.annotate(
                shorten_label(block),
                (x_values[index], y_values[index]),
                fontsize=8,
                alpha=0.8,
            )

    plt.tight_layout()
    plt.show()


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Generate text with local embeddings and visualize block embeddings."
    )
    parser.add_argument("input_file", help="Path to a .txt training file")
    parser.add_argument(
        "--block-size",
        type=int,
        default=20,
        help="Number of tokens in each embedded context block, default: 20",
    )
    parser.add_argument(
        "--next-size",
        type=int,
        default=5,
        help="Number of continuation tokens copied per generation step, default: 5",
    )
    parser.add_argument(
        "--sentences",
        type=int,
        default=1,
        help="Number of outputs to generate, default: 1",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=80,
        help="Maximum tokens per generated output, default: 80",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Randomly choose among this many nearest blocks, default: 5",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for repeatable output",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name, default: all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open an interactive 2D scatter chart of the block embeddings",
    )
    parser.add_argument(
        "--label-points",
        action="store_true",
        help="Label points with shortened block text snippets",
    )
    parser.add_argument(
        "--max-labels",
        type=int,
        default=30,
        help="Maximum number of point labels to draw, default: 30",
    )
    return parser.parse_args(argv)


def validate_args(args):
    if args.block_size < 1:
        return "--block-size must be at least 1."
    if args.next_size < 1:
        return "--next-size must be at least 1."
    if args.sentences < 1:
        return "--sentences must be at least 1."
    if args.max_words < 1:
        return "--max-words must be at least 1."
    if args.top_k < 1:
        return "--top-k must be at least 1."
    if args.max_labels < 0:
        return "--max-labels cannot be negative."
    return None


def main(argv=None):
    args = parse_args(argv)
    validation_error = validate_args(args)
    if validation_error:
        print(f"Error: {validation_error}", file=sys.stderr)
        return 2

    try:
        text = load_text(args.input_file)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if not text.strip():
        print("Error: input file is empty.", file=sys.stderr)
        return 1

    tokens = tokenize(text)
    if not tokens:
        print("Error: input file does not contain usable words.", file=sys.stderr)
        return 1

    if len(tokens) <= args.block_size:
        fallback = join_tokens(tokens)
        if fallback and fallback[-1] not in SENTENCE_ENDINGS:
            fallback += "."
        print(fallback)
        return 0

    blocks, continuations = make_blocks(tokens, args.block_size, args.next_size)
    if not blocks:
        print("Error: could not create training blocks from the input.", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)

    try:
        model = load_embedding_model(args.model)
        embeddings = embed_blocks(model, blocks)
        normalized_embeddings = normalize_embeddings(embeddings)
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if args.visualize:
        try:
            points = project_embeddings(embeddings, args.seed)
            plot_embeddings(
                points,
                blocks,
                args.model,
                args.block_size,
                args.label_points,
                args.max_labels,
            )
        except (RuntimeError, ValueError) as error:
            print(f"Error: {error}", file=sys.stderr)
            return 1

    generated = 0
    for _ in range(args.sentences):
        sentence = generate_sentence(
            model,
            blocks,
            continuations,
            normalized_embeddings,
            args.block_size,
            args.max_words,
            args.top_k,
            rng,
        )
        if sentence:
            print(sentence)
            generated += 1

    if generated == 0:
        print("Error: could not generate text from the input.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
