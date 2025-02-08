import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

# Ensure necessary NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')


def read_article(file_name: str) -> list:
    """Reads the article from a given file and tokenizes it into sentences."""
    try:
        with open(file_name, "r") as file:
            filedata = file.readlines()

        if not filedata:  # Handle empty files
            print(f"Error: The file '{file_name}' is empty.")
            return []

        article = " ".join(filedata)  # Join all lines in case the file spans multiple lines
        sentences = nltk.sent_tokenize(article)  # Use NLTK's sentence tokenizer

        # Tokenize each sentence into words
        sentences = [sentence.split(" ") for sentence in sentences]
        return sentences

    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found.")
        return []


def sentence_similarity(sent1: list, sent2: list, stopwords: set = None) -> float:
    """Calculates the cosine similarity between two sentences."""
    if stopwords is None:
        stopwords = set()

    # Convert sentences to lowercase and remove stopwords
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))

    # Create vectors for the two sentences
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    similarity = 1 - cosine_distance(vector1, vector2)
    if np.isnan(similarity):  # Handle zero vectors
        return 0
    return similarity


def gen_sim_matrix(sentences: list, stop_words: set) -> np.ndarray:
    """Generates a similarity matrix for the sentences."""
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_summary(file_name: str, summary_size: str = 'medium') -> str:
    """Generates the summary of the text in the given file.
    
    Args:
        file_name: Path to the text file
        summary_size: Size of summary ('short', 'medium', or 'detailed')
    """
    stop_words = set(stopwords.words('english'))
    summarize_text = []

    sentences = read_article(file_name)

    if not sentences:
        return "Error: No valid sentences found in the file."

    total_sentences = len(sentences)
    
    # Define ratios based on summary size
    size_ratios = {
        'short': 0.3,    # 30% of original text
        'medium': 0.5,   # 50% of original text
        'detailed': 0.7  # 70% of original text
    }
    
    # Get ratio based on selected size (default to medium if invalid size provided)
    ratio = size_ratios.get(summary_size, 0.5)
    
    # Calculate number of sentences for summary
    # Ensure at least 3 sentences for short, 5 for medium, 7 for detailed
    min_sentences = {'short': 3, 'medium': 5, 'detailed': 7}
    min_sent = min_sentences.get(summary_size, 5)
    
    # Calculate final number of sentences
    top_n = max(min_sent, int(total_sentences * ratio))
    top_n = min(top_n, total_sentences)  # Don't exceed total sentences

    sentence_similarity_matrix = gen_sim_matrix(sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    # Get the top N sentences for the summary
    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))

    return ". ".join(summarize_text)
