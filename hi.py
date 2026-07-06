import random
import re
from collections import defaultdict


TRAINING_TEXT = """
A graphics processing unit (GPU) is a specialized electronic circuit designed for digital image processing and to accelerate computer graphics, being present either as a component on a discrete graphics card or embedded on motherboards, mobile phones, personal computers, workstations, and game consoles. GPUs are also increasingly being used for artificial intelligence (AI) processing due to linear algebra acceleration, which is also used extensively in graphics processing.

Although there is no single definition of the term, and it may be used to describe any video display system, in modern use a GPU includes the ability to internally perform the calculations needed for various graphics tasks, like rotating and scaling 3D images, and often the additional ability to run custom programs known as shaders. This contrasts with earlier graphics controllers known as video display controllers which had no internal calculation capabilities, or blitters, which performed only basic memory movement operations. The modern GPU emerged during the 1990s, adding the ability to perform operations like drawing lines and text without CPU help, and later adding 3D functionality.

Graphics functions are generally independent and this lends these tasks to being implemented on separate calculation engines. Modern GPUs include hundreds, or thousands, of calculation units. This made them useful for non-graphic calculations involving embarrassingly parallel problems due to their parallel structure. The ability of GPUs to rapidly perform vast numbers of calculations has led to their adoption in diverse fields including artificial intelligence (AI) where they excel at handling data-intensive and computationally demanding tasks. Other non-graphical uses include the training of neural networks and cryptocurrency mining.

GPU companies
Main article: List of graphics chips and card companies
Many companies have produced GPUs under a number of brand names. In 2009,[needs update] Intel, Nvidia, and AMD/ATI were the market share leaders, with 49.4%, 27.8%, and 20.6% market share respectively. In addition, Matrox[1] while originally producing custom solutions, now customizes GPUs from Intel and AMD for workstation usage. Chinese companies such as Jingjia Micro have also produced GPUs for the domestic market although in terms of worldwide sales, they lag behind market leaders.[2]

Computational functions

The ATI HD5470 GPU (above, with copper heatpipe attached) features UVD 2.1 which enables it to decode AVC and VC-1 video formats.
Several factors of GPU construction affect the performance of the card for real-time rendering, such as the size of the connector pathways in the semiconductor device fabrication, the clock signal frequency, and the number and size of various on-chip memory caches. Performance is also affected by the number of streaming multiprocessors (SM) for NVidia GPUs, or compute units (CU) for AMD GPUs, or Xe cores for Intel Xe-based GPUs, which describe the number of on-silicon processor core units within the GPU chip that perform the core calculations, typically working in parallel with other SM/CUs on the GPU. GPU performance is typically measured in floating point operations per second (FLOPS); Modern GPUs typically deliver performance measured in teraflops (TFLOPS). This is an estimated performance measure, and should not be treated as fact, as other factors can affect actual performance.[3]

Modern GPUs also include dedicated hardware blocks for ray tracing, video encoding, and AI acceleration.

GPU forms
In personal computers, there are two main forms of GPUs: dedicated graphics (also called discrete graphics) and integrated graphics (also called shared graphics solutions, integrated graphics processors (IGP), or unified memory architecture (UMA). [4]

Dedicated graphics processing unit
See also: Video card
Dedicated graphics processing units use on board RAM that is dedicated to the GPU rather than relying on the computer's main system memory. This RAM is usually specially selected for the expected serial workload of the graphics card, such as GDDR SDRAM. This has massive performance benefits, but the caveat of "choking" when running out of dedicated memory, worsening performance.

Technologies such as Scalable Link Interface (SLI), NVLink, and CrossFire allow multiple GPUs to draw images simultaneously for a single screen, increasing the processing power available for graphics. These technologies, however, are increasingly uncommon; most games do not fully use multiple GPUs, as most users cannot afford them.[5][6][7][better source needed] Multiple GPUs are still used on supercomputers (such as in Summit); on workstations to accelerate video (processing multiple videos at once)[8][4][9] and 3D rendering;[10] for visual effects (VFX);[11] general purpose graphics processing unit (GPGPU) workloads and for simulations,[12] and in AI to expedite training, as is the case with Nvidia's lineup of DGX workstations and servers.[citation needed]
"""


SENTENCE_ENDINGS = {".", "!", "?"}


def tokenize(text):
    """Split text into words and sentence-ending punctuation."""
    return re.findall(r"\w+(?:['-]\w+)*|[.!?]", text)


def build_chain(words):
    """Map each word to the words that appeared directly after it."""
    chain = defaultdict(list)

    for current_word, next_word in zip(words, words[1:]):
        chain[current_word].append(next_word)

    return chain


def find_sentence_boundaries(words):
    """Find words that start sentences and words that end sentences."""
    starting_words = []
    ending_words = set()

    if words and words[0] not in SENTENCE_ENDINGS:
        starting_words.append(words[0])

    for index, word in enumerate(words):
        if word not in SENTENCE_ENDINGS:
            continue

        if index > 0 and words[index - 1] not in SENTENCE_ENDINGS:
            ending_words.add(words[index - 1])
        if index + 1 < len(words) and words[index + 1] not in SENTENCE_ENDINGS:
            starting_words.append(words[index + 1])

    return starting_words, ending_words


def join_words(words):
    """Join generated words while attaching punctuation to the previous word."""
    sentence = ""
    for word in words:
        if word in SENTENCE_ENDINGS:
            sentence = sentence.rstrip() + word + " "
        else:
            sentence += word + " "
    return sentence.strip()


def generate_sentence(chain, start_word, ending_words, length=30, minimum_length=8):
    """Generate text by repeatedly choosing one learned next word."""
    if start_word not in chain:
        raise ValueError(f"The start word {start_word!r} was not found in the text.")

    words = [start_word]
    current_word = start_word

    for _ in range(length - 1):
        possible_next_words = chain.get(current_word)
        if not possible_next_words:
            break

        next_word = random.choice(possible_next_words)
        words.append(next_word)
        current_word = next_word

        if next_word in ending_words and len(words) >= minimum_length:
            break

    if words[-1] not in SENTENCE_ENDINGS:
        words.append(".")

    return join_words(words)


def main():
    words = tokenize(TRAINING_TEXT)
    chain = build_chain(words)
    starting_words, ending_words = find_sentence_boundaries(words)

    start_word = random.choice(starting_words)
    sentence = generate_sentence(chain, start_word, ending_words, length=40)
    print(sentence)


if __name__ == "__main__":
    main()
