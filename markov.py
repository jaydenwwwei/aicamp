import argparse
import random
import re
import sys
from collections import defaultdict


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


def build_chain(tokens, order):
    """Build a Markov chain mapping word tuples to possible next tokens."""
    chain = defaultdict(list)

    if len(tokens) <= order:
        return chain

    for index in range(len(tokens) - order):
        state = tuple(tokens[index:index + order])
        next_token = tokens[index + order]
        chain[state].append(next_token)

    return chain


def join_tokens(tokens):
    """Join tokens while keeping punctuation attached to the previous word."""
    sentence = ""
    for token in tokens:
        if token in SENTENCE_ENDINGS:
            sentence = sentence.rstrip() + token + " "
        else:
            sentence += token + " "
    return sentence.strip()


def choose_start_state(chain, rng):
    sentence_starts = [
        state for state in chain
        if state[0] not in SENTENCE_ENDINGS
    ]
    if not sentence_starts:
        return None

    likely_starts = [
        state for state in sentence_starts
        if state[0][:1].isupper()
    ]
    return rng.choice(likely_starts or sentence_starts)


def generate_sentence(chain, order, rng, max_words=40):
    """Generate one sentence-like string from a Markov chain."""
    if not chain:
        return ""

    state = choose_start_state(chain, rng)
    if state is None:
        return ""

    output = list(state)

    for _ in range(max_words):
        choices = chain.get(state)
        if not choices:
            break

        next_token = rng.choice(choices)
        output.append(next_token)

        if next_token in SENTENCE_ENDINGS:
            break

        state = tuple(output[-order:])

    if output[-1] not in SENTENCE_ENDINGS:
        output.append(".")

    return join_tokens(output)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Generate sentence-like text using a Markov chain."
    )
    parser.add_argument("input_file", help="Path to a .txt training file")
    parser.add_argument(
        "--order",
        type=int,
        default=2,
        help="Markov chain order, default: 2",
    )
    parser.add_argument(
        "--sentences",
        type=int,
        default=1,
        help="Number of sentences to generate, default: 1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for repeatable output",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.order < 1:
        print("Error: --order must be at least 1.", file=sys.stderr)
        return 2
    if args.sentences < 1:
        print("Error: --sentences must be at least 1.", file=sys.stderr)
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

    if len(tokens) <= args.order:
        fallback = join_tokens(tokens)
        if fallback and fallback[-1] not in SENTENCE_ENDINGS:
            fallback += "."
        print(fallback)
        return 0

    chain = build_chain(tokens, args.order)
    rng = random.Random(args.seed)

    generated = 0
    for _ in range(args.sentences):
        sentence = generate_sentence(chain, args.order, rng)
        if sentence:
            print(sentence)
            generated += 1

    if generated == 0:
        print("Error: could not generate text from the input.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
