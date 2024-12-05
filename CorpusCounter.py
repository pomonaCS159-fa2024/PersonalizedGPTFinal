import json
from collections import Counter

def load_json(file_path):
   
    with open(file_path, 'r') as file:
        return json.load(file)

def count_words(data, stop_words):
   
    word_counter = Counter()
    for entry in data:
        text = entry.get('prompt', '') + " " + entry.get('response', '')
        words = text.lower().split()
        filtered_words = [word.strip('.,!?') for word in words if word not in stop_words]
        word_counter.update(filtered_words)
    return word_counter

def display_top_words(word_counter, top_n=20):
    print(f"Top {top_n} most frequently used words:")
    for word, count in word_counter.most_common(top_n):
        print(f"{word}: {count}")

if __name__ == "__main__":
  
    json_file_path = "/Users/avga2021/Desktop/NLP_Final2/NLPData-3.json"
    
    # List of stop words to exclude
    stop_words = {"the", "and", "a", "it", "is", "of", "to", "in", "for", "on", "that", "i","you","but","it","me","it's","are","i'm","they"}
    

    data = load_json(json_file_path)

    word_frequencies = count_words(data, stop_words)
    
    # Display top 10 most frequently used words
    display_top_words(word_frequencies)
