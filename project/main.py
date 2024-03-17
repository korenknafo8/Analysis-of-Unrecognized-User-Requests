import json
import pandas as pd
import numpy as np
from compare_clustering_solutions import evaluate_clustering
from sentence_transformers import SentenceTransformer
# full documentation - https://huggingface.co/sentence-transformers
MODEL_NAME = 'all-MiniLM-L6-v2'
model = SentenceTransformer(MODEL_NAME)
TRESH_VAL = 1
def euclidean_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)

def embed_sentences(sentences):
    embedding_arr = []
    for sentence in sentences:
        embedding_arr.append(model.encode(sentence))

    return embedding_arr


def sentences_clustering(data, threshold, min_size):
    groups = {}  # Initialize groups dictionary
    dict_counter = 0
    
    for vector in data:
        min_group_key = None
        min_distance = TRESH_VAL  # Initialize minimum distance to a large value
        
        # Find the closest group
        for key, group in groups.items():
            distance = euclidean_distance(group[0], vector)
            if distance < min_distance:
                min_distance = distance
                min_group_key = key
        
        # If no group found within threshold or if no groups exist
        if min_group_key is None or min_distance > threshold:
            dict_counter += 1
            groups[dict_counter] = [vector, 1]
        else:
            # Update existing group
            min_group = groups[min_group_key]
            min_group[0] = (min_group[0] * min_group[1] + vector) / (min_group[1] + 1)
            min_group[1] += 1
    
    # Remove groups with size less than min_group_size
    groups = {key: value for key, value in groups.items() if value[1] >= min_size}

    return groups

                

def analyze_unrecognized_requests(data_file, output_file, min_size):
    # todo: implement this function
    #  you are encouraged to break the functionality into multiple functions,
    #  but don't split your code into multiple *.py files
    #
    #  todo: the final outcome is the json file with clustering results saved as output_file

    # Read data from CSV file
    data = pd.read_csv(data_file)
    text_data = data["text"]
    
    # Assuming you have a function embed_sentences that embeds the sentences into vectors
    embedded_vectors = embed_sentences(text_data)
    
    # Perform clustering on embedded vectors
    clusters = sentences_clustering(embedded_vectors, threshold=2, min_size=int(min_size))
    
    print(clusters.keys())
    


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['min_cluster_size'])

    # todo: evaluate your clustering solution against the provided one
    evaluate_clustering(config['example_solution_file'], config['example_solution_file'])  # invocation example
    #evaluate_clustering(config['example_solution_file'], config['output_file'])
