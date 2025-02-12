import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
from collections import Counter

# Ensure necessary NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')


def preprocess_text(sentences: list) -> list:
    """Preprocess sentences while keeping short but meaningful ones."""
    processed_sentences = []

    for sentence in sentences:
        cleaned = ' '.join(sentence).strip()

        # Keep short but meaningful sentences
        if len(cleaned.split()) < 3:
            continue

        cleaned = cleaned.replace('"', '').replace("'", "")
        processed_sentences.append(cleaned.split())

    return processed_sentences


def read_article(file_name: str) -> list:
    """Reads the article and tokenizes it into sentences."""
    try:
        with open(file_name, "r") as file:
            filedata = file.readlines()

        if not filedata:
            print(f"Error: The file '{file_name}' is empty.")
            return []

        article = " ".join(filedata)
        sentences = nltk.sent_tokenize(article)
        sentences = [sentence.split(" ") for sentence in sentences]

        return preprocess_text(sentences)

    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found.")
        return []


def sentence_similarity(sent1: list, sent2: list, stopwords: set = None) -> float:
    """Improved sentence similarity calculation using word uniqueness."""
    if stopwords is None:
        stopwords = set()

    sent1 = [w.lower() for w in sent1 if w.lower() not in stopwords]
    sent2 = [w.lower() for w in sent2 if w.lower() not in stopwords]

    unique_words = set(sent1 + sent2)
    word_frequencies = Counter(sent1 + sent2)

    vector1 = [1 / word_frequencies[word] for word in unique_words]  # Less frequent words get higher weight
    vector2 = [1 / word_frequencies[word] for word in unique_words]

    similarity = 1 - cosine_distance(vector1, vector2)
    return max(0, similarity)


def gen_sim_matrix(sentences: list, stop_words: set) -> np.ndarray:
    """Generates a similarity matrix for ranking sentences."""
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def get_most_important_sentences(sentences: list, num_sentences: int) -> list:
    """Selects the most important sentences while maintaining logical flow dynamically."""

    # Count word frequency across all sentences
    word_frequencies = Counter([word.lower() for sentence in sentences for word in sentence])

    # Rank sentences based on overall importance (word frequency + sentence position weight)
    ranked_sentences = sorted(
        enumerate(sentences),
        key=lambda x: (
                sum(word_frequencies[word.lower()] for word in x[1]) +
                (5 if x[0] == 0 else 0) +  # Extra weight for the first sentence
                (3 if x[0] == len(sentences) - 1 else 0)  # Extra weight for the last sentence
        ),
        reverse=True
    )

    # Select sentences while ensuring we don't exceed num_sentences
    selected_indices = []
    for index, sentence in ranked_sentences:
        if len(selected_indices) >= num_sentences:
            break
        selected_indices.append(index)

    return sorted(selected_indices)


def generate_summary(file_name: str, summary_size: str = 'medium') -> str:
    """Generates a structured summary for any type of story while maintaining readability."""
    stop_words = set(stopwords.words('english'))
    sentences = read_article(file_name)

    if not sentences:
        return "Error: No valid sentences found in the file."

    total_sentences = len(sentences)

    # Summary size selection
    size_ratios = {'short': 0.3, 'medium': 0.5, 'detailed': 0.7}
    min_sentences = {'short': 2, 'medium': 4, 'detailed': 6}

    ratio = size_ratios.get(summary_size, 0.5)
    min_sent = min_sentences.get(summary_size, 4)

    # Calculate the number of sentences for summary while ensuring 30%, 50%, and 70% differences
    top_n = max(1, min(int(total_sentences * ratio), total_sentences - 1))

    # Generate similarity matrix
    similarity_matrix = gen_sim_matrix(sentences, stop_words)

    # Create graph and calculate scores
    sentence_similarity_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Get ranked sentences with their original positions
    ranked_sentences = sorted([(scores[i], i, s) for i, s in enumerate(sentences)], reverse=True)

    # Ensure logical flow: Always keep the first and last sentence
    key_sentences = [0]
    if len(sentences) > 1:
        key_sentences.append(len(sentences) - 1)

        # Limit the number of selected sentences to exactly top_n
    selected_indices = sorted(set(key_sentences))
    for _, idx, _ in ranked_sentences:
        if len(selected_indices) < top_n:
            selected_indices.append(idx)
        else:
            break

            # Reconstruct summary while maintaining readability
    structured_summary = []
    for idx in sorted(set(selected_indices)):
        structured_summary.append(" ".join(sentences[idx]))

    return "\n\n".join(structured_summary)  # Adds spacing for better readability
