import random
import re

import streamlit as st

from markov import SENTENCE_ENDINGS, build_chain, generate_sentence, join_tokens, tokenize


FALLBACK_CORPUS = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.

However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters.

My dear Mr. Bennet, said his lady to him one day, have you heard that Netherfield Park is let at last?

Mr. Bennet replied that he had not.

But it is, returned she; for Mrs. Long has just been here, and she told me all about it.

Mr. Bennet made no answer.

Do you not want to know who has taken it? cried his wife impatiently.

You want to tell me, and I have no objection to hearing it.

This was invitation enough.

Why, my dear, you must know, Mrs. Long says that Netherfield is taken by a young man of large fortune from the north of England; that he came down on Monday in a chaise and four to see the place, and was so much delighted with it, that he agreed with Mr. Morris immediately.

What is his name?

Bingley.

Is he married or single?

Oh! single, my dear, to be sure! A single man of large fortune; four or five thousand a year. What a fine thing for our girls!

How so? how can it affect them?

My dear Mr. Bennet, replied his wife, how can you be so tiresome! You must know that I am thinking of his marrying one of them.

Is that his design in settling here?

Design! nonsense, how can you talk so! But it is very likely that he may fall in love with one of them, and therefore you must visit him as soon as he comes.
"""


PROFESSOR_PICKLE_CORPUS = """According to my sandwich research, the moon has misplaced its academic hat.
I must object with great scholarly confusion and a spoon full of mustard.
The evidence suggests that every toaster contains a tiny philosopher wearing socks.
Behold my clipboard of destiny, for it contains three pickles and a suspicious diagram.
In conclusion, breakfast is not a meal but a committee of crumbs.
My dear colleague, your argument is bold, crunchy, and probably illegal in several libraries.
I have calculated the emotional velocity of soup, and the answer is Thursday.
Please remain calm while I consult the encyclopedia of dramatic vegetables.
No serious scholar can ignore the ancient connection between cheese and thunder.
The laboratory mice have unionized, and frankly their demands are reasonable.
"""


CAPTAIN_WAFFLE_CORPUS = """Ahoy, behold the breakfast prophecy and tighten the syrup cannon.
The waffles have voted, and the banana republic is nervous.
Kaboom, my friend, the pancake engine has achieved maximum nonsense.
I challenge the moon to a duel at brunch, with extra butter and no regrets.
Spin the wheel of toast, because destiny smells like maple thunder.
Captain Waffle never retreats unless the coffee machine starts singing opera.
The rubber duck navy is prepared, the spatulas are polished, and chaos is on sale.
Raise the flag of crispy confusion and sail toward the gravy horizon.
By the power of breakfast, I declare this conversation officially crunchy.
Do not fear the jelly storm, for I brought emergency muffins.
"""


BANANA_CREW_CORPUS = """Banana boss says the plan is simple, then forgets the plan inside a lunchbox.
Tiny goggles shine under the lab lights while everyone cheers for the wrong button.
The rocket is ready, the banana is missing, and the floor is covered in giggles.
One little helper says hello to the mop, salutes the ceiling, and falls into a basket.
Nobody knows why the machine is dancing, but the machine seems very confident.
The crew marches proudly into the room, turns around, and marches proudly back out.
Banana snacks are serious business, especially when the alarm clock starts wearing shoes.
The villain speech begins with thunder, but ends when somebody slips on pudding.
Every invention needs science, courage, and a small yellow friend yelling hooray.
The mission is secret, chaotic, and mostly about finding more bananas.
"""


GAMING_STREAM_CORPUS = """Chat, listen, that round was unbelievable and I need everyone to breathe for two seconds.
No way, no way, that timing was actually perfect and the whole lobby knows it.
We are locked in right now, the headset is on, the chair is moving, and the vibes are dangerous.
Somebody clip that because the play looked impossible until it became history.
I am not yelling, I am explaining strategy at competitive volume.
The team needs one clean call, one good swing, and absolutely no panic from the snack department.
That was almost genius, almost tragic, and somehow still worth watching again.
Chat is saying relax, but the scoreboard is saying dramatic comeback arc.
If this works, we celebrate; if it fails, we call it experimental content.
The controller is innocent, the plan was brave, and the replay will judge us all.
"""


SINGLE_CHATBOT_CORPUSES = {
    "Pride and Prejudice fallback": FALLBACK_CORPUS,
    "Goofy banana comedy sample": BANANA_CREW_CORPUS,
    "Gaming stream banter sample": GAMING_STREAM_CORPUS,
    "Custom pasted text": "",
}


def sanitize_text(text):
    """Remove transcript timestamps and metadata that pollute the Markov chain."""
    text = re.sub(r"<system-reminder>.*?</system-reminder>", " ", text, flags=re.DOTALL)
    text = re.sub(r"\b\d{1,2}:\d{2}:\d{2}\b", " ", text)
    text = re.sub(r"\b\d{1,2}:\d{2}\b", " ", text)
    text = re.sub(
        r"\b\d+\s+hours?,\s*\d+\s+minutes?,\s*\d+\s+seconds?\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b\d+\s+minutes?,\s*\d+\s+seconds?\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b\d+\s+seconds?\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text.strip()


def generate_from_text(text, order, sentence_count, max_words, seed):
    tokens = tokenize(text)
    if not tokens:
        return [], "The training text does not contain usable words."

    if len(tokens) <= order:
        fallback = join_tokens(tokens)
        if fallback and fallback[-1] not in SENTENCE_ENDINGS:
            fallback += "."
        return [fallback], "The text is shorter than the selected order, so the app returned the input text instead."

    chain = build_chain(tokens, order)
    if not chain:
        return [], "Could not build a Markov chain from this text."

    rng = random.Random(seed)
    sentences = []
    for _ in range(sentence_count):
        sentence = generate_sentence(chain, order, rng, max_words=max_words)
        if sentence:
            sentences.append(sentence)

    if not sentences:
        return [], "Could not generate any sentences from this text."

    return sentences, None


def generate_bot_conversation(bot_one_text, bot_two_text, order, turns, max_words, seed):
    bot_one_rng = random.Random(None if seed is None else seed)
    bot_two_rng = random.Random(None if seed is None else seed + 1000)
    bot_one_chain = build_chain(tokenize(sanitize_text(bot_one_text)), order)
    bot_two_chain = build_chain(tokenize(sanitize_text(bot_two_text)), order)

    if not bot_one_chain or not bot_two_chain:
        return [], "Both bots need enough training text to build a Markov chain."

    messages = []
    for turn in range(turns):
        if turn % 2 == 0:
            name = "Banana Crew"
            chain = bot_one_chain
            rng = bot_one_rng
        else:
            name = "Stream Goblin"
            chain = bot_two_chain
            rng = bot_two_rng

        message = generate_sentence(chain, order, rng, max_words=max_words)
        if message:
            messages.append({"name": name, "content": message})

    if not messages:
        return [], "The bots could not generate a conversation."

    return messages, None


def main():
    st.set_page_config(page_title="Markov Chatbot", page_icon="M", layout="wide")

    st.title("Markov Chain Chatbot")
    st.write("Chat with a local Markov bot. Paste training text, or leave it empty to use Pride and Prejudice as the fallback corpus.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "reply_count" not in st.session_state:
        st.session_state.reply_count = 0
    if "bot_messages" not in st.session_state:
        st.session_state.bot_messages = []

    with st.sidebar:
        st.header("Settings")
        mode = st.radio("Mode", ["Single chatbot", "Two funny bots"])
        order = st.number_input("Markov order", min_value=1, max_value=20, value=2, step=1)
        max_words = st.number_input("Max words per reply", min_value=5, max_value=200, value=40, step=5)
        turns = 10
        if mode == "Two funny bots":
            turns = st.number_input("Conversation turns", min_value=2, max_value=50, value=10, step=2)
        use_seed = st.checkbox("Use random seed")
        seed = None
        if use_seed:
            seed = st.number_input("Seed", min_value=0, value=42, step=1)
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.session_state.bot_messages = []
            st.session_state.reply_count = 0
            st.rerun()

    if mode == "Two funny bots":
        st.subheader("Two Funny Bots")
        st.write("Banana Crew and Stream Goblin take turns talking using separate Markov chains.")

        with st.expander("Customize bot corpuses"):
            bot_one_text = st.text_area(
                "Banana Crew corpus",
                value=BANANA_CREW_CORPUS,
                height=180,
            )
            bot_two_text = st.text_area(
                "Stream Goblin corpus",
                value=GAMING_STREAM_CORPUS,
                height=180,
            )

        if st.button("Start bot conversation", type="primary"):
            messages, warning = generate_bot_conversation(
                bot_one_text,
                bot_two_text,
                int(order),
                int(turns),
                int(max_words),
                None if seed is None else int(seed),
            )
            if warning:
                st.warning(warning)
            st.session_state.bot_messages = messages

        for message in st.session_state.bot_messages:
            with st.chat_message("assistant"):
                st.markdown(f"**{message['name']}:** {message['content']}")

        return

    corpus_choice = st.selectbox("Training source", list(SINGLE_CHATBOT_CORPUSES.keys()))
    selected_corpus = SINGLE_CHATBOT_CORPUSES[corpus_choice]

    sample_text = st.text_area(
        "Optional training text",
        value=selected_corpus,
        height=220,
        placeholder="Paste text here, or leave empty to use Pride and Prejudice.",
        help="Choose a built-in sample or paste your own text. If empty, the app uses Pride and Prejudice.",
    )

    raw_training_text = sample_text.strip() or FALLBACK_CORPUS
    training_text = sanitize_text(raw_training_text)
    using_fallback = not sample_text.strip()
    raw_token_count = len(tokenize(raw_training_text))
    token_count = len(tokenize(training_text))

    if using_fallback:
        st.info("Using fallback corpus: Pride and Prejudice.")
    elif corpus_choice != "Custom pasted text":
        st.info(f"Using built-in training sample: {corpus_choice}.")

    if raw_training_text != training_text:
        st.caption(f"Detected tokens after cleanup: {token_count} removed/changed from raw count: {raw_token_count}")
    else:
        st.caption(f"Detected tokens: {token_count}")

    st.subheader("Chat")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Send a message")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        reply_seed = None
        if seed is not None:
            reply_seed = int(seed) + st.session_state.reply_count

        sentences, warning = generate_from_text(
            training_text,
            int(order),
            1,
            int(max_words),
            reply_seed,
        )
        reply = sentences[0] if sentences else "I could not generate a reply from that training text."

        with st.chat_message("assistant"):
            if warning:
                st.warning(warning)
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.session_state.reply_count += 1


if __name__ == "__main__":
    main()
