"""
orthowm Project: Orthographic Neighborhood Density Analysis
Generates 4-letter English nouns and calculates their orthographic neighborhood
density using Levenshtein distance, then categorizes into high/low density groups.
"""

import nltk
from nltk.corpus import wordnet, words
from collections import defaultdict
import numpy as np

# Download required NLTK data
nltk.download('wordnet')
nltk.download('words')


def get_4_letter_nouns():
    """
    Extract 4-letter English nouns from WordNet.
    Returns a list of unique 4-letter nouns.
    """
    nouns = set()
    
    # Get all nouns from WordNet
    for synset in wordnet.all_synsets('n'):
        for lemma in synset.lemmas():
            word = lemma.name().lower()
            # Filter for 4-letter words containing only alphabetic characters
            if len(word) == 4 and word.isalpha():
                nouns.add(word)
    
    return sorted(list(nouns))


def levenshtein_distance(s1, s2):
    """
    Calculate the Levenshtein distance between two strings.
    Uses dynamic programming approach.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def calculate_neighborhood_density(word_list, distance_threshold=1):
    """
    Calculate orthographic neighborhood density for each word.
    Neighborhood density = number of words at Levenshtein distance <= threshold
    """
    density_dict = {}
    
    for i, word1 in enumerate(word_list):
        neighbors = 0
        for j, word2 in enumerate(word_list):
            if i != j:
                dist = levenshtein_distance(word1, word2)
                if dist <= distance_threshold:
                    neighbors += 1
        density_dict[word1] = neighbors
    
    return density_dict


def categorize_by_density(density_dict):
    """
    Categorize words into high and low density groups based on median split.
    """
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


def main():
    print("=" * 60)
    print("Orthographic Neighborhood Density Analysis")
    print("=" * 60)
    
    # Step 1: Get 4-letter nouns
    print("\n[Step 1] Extracting 4-letter English nouns from WordNet...")
    nouns = get_4_letter_nouns()
    print(f"Found {len(nouns)} 4-letter nouns")
    
    # Step 2: Select first 100 (or all if less than 100)
    sample_size = min(100, len(nouns))
    word_list = nouns[:sample_size]
    print(f"Using {sample_size} words for analysis")
    
    # Step 3: Calculate neighborhood density
    print("\n[Step 2] Calculating orthographic neighborhood density...")
    print("(This may take a moment as it compares all word pairs)")
    density_dict = calculate_neighborhood_density(word_list, distance_threshold=1)
    
    # Step 4: Categorize
    print("\n[Step 3] Categorizing by density...")
    high_density, low_density, median = categorize_by_density(density_dict)
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nMedian neighborhood density: {median:.1f}")
    print(f"High density group: {len(high_density)} words")
    print(f"Low density group: {len(low_density)} words")
    
    print("\n" + "-" * 60)
    print("HIGH DENSITY WORDS (neighbors >= median)")
    print("-" * 60)
    for word, density in high_density.items():
        print(f"{word}: {density} neighbors")
    
    print("\n" + "-" * 60)
    print("LOW DENSITY WORDS (neighbors < median)")
    print("-" * 60)
    for word, density in low_density.items():
        print(f"{word}: {density} neighbors")
    
    # Save to file
    print("\n" + "=" * 60)
    print("Saving results to 'neighborhood_density_results.txt'...")
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
    
    print("Results saved successfully!")
    
    return word_list, density_dict, high_density, low_density


if __name__ == "__main__":
    main()
