"""
orthowm Project: Orthographic Neighborhood Density Analysis
Generates 4-letter English nouns and calculates their orthographic neighborhood
density using Levenshtein distance, then categorizes into high/low density groups.
"""

import logging
import nltk
from nltk.corpus import wordnet, words
from collections import defaultdict
import numpy as np
from rapidfuzz.distance import Levenshtein

logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download('wordnet')
nltk.download('words')


def get_4_letter_nouns() -> list[str]:
    nouns = set()
    
    # Get all nouns from WordNet
    for synset in wordnet.all_synsets('n'):
        for lemma in synset.lemmas():
            word = lemma.name().lower()
            # Filter for 4-letter words containing only alphabetic characters
            if len(word) == 4 and word.isalpha():
                nouns.add(word)
    
    return sorted(list(nouns))




def calculate_neighborhood_density(word_list: list[str], distance_threshold: int = 1) -> dict[str, int]:
    density_dict = {}
    
    for i, word1 in enumerate(word_list):
        neighbors = 0
        for j, word2 in enumerate(word_list):
            if i != j:
                dist = Levenshtein.distance(word1, word2)
                if dist <= distance_threshold:
                    neighbors += 1
        density_dict[word1] = neighbors
    
    return density_dict


def categorize_by_density(density_dict: dict[str, int]) -> tuple[dict[str, int], dict[str, int], float]:
    densities = list(density_dict.values())
    median_density = np.median(densities)
    
    high_density = {word: density for word, density in density_dict.items() 
                    if density >= median_density}
    low_density = {word: density for word, density in density_dict.items() 
                   if density < median_density}
    
    # Sort by density (descending for high, ascending for low)
    high_density = dict(sorted(high_density.items(), key=lambda x: x[1], reverse=True))
    low_density = dict(sorted(low_density.items(), key=lambda x: x[1]))
    
    return high_density, low_density, median_density


def main() -> tuple[list[str], dict[str, int], dict[str, int], dict[str, int]]:
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Extracting 4-letter English nouns from WordNet...")
    nouns = get_4_letter_nouns()
    logger.info(f"Found {len(nouns)} 4-letter nouns")
    
    sample_size = min(100, len(nouns))
    word_list = nouns[:sample_size]
    logger.info(f"Using {sample_size} words for analysis")
    
    logger.info("Calculating orthographic neighborhood density...")
    density_dict = calculate_neighborhood_density(word_list, distance_threshold=1)
    
    high_density, low_density, median = categorize_by_density(density_dict)
    
    logger.info(f"Median neighborhood density: {median:.1f}")
    logger.info(f"High density group: {len(high_density)} words")
    logger.info(f"Low density group: {len(low_density)} words")
    
    with open('neighborhood_density_results.txt', 'w') as f:
        f.write("Orthographic Neighborhood Density Analysis Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total words analyzed: {sample_size}\n")
        f.write(f"Median neighborhood density: {median:.1f}\n\n")
        
        f.write("HIGH DENSITY GROUP (neighbors >= median)\n")
        f.write("-" * 60 + "\n")
        for word, density in high_density.items():
            f.write(f"{word}: {density} neighbors\n")
        
        f.write("\nLOW DENSITY GROUP (neighbors < median)\n")
        f.write("-" * 60 + "\n")
        for word, density in low_density.items():
            f.write(f"{word}: {density} neighbors\n")
    
    logger.info("Results saved successfully!")
    
    return word_list, density_dict, high_density, low_density


if __name__ == "__main__":
    main()
