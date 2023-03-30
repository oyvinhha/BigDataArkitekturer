# This is the code for the LSH project of TDT4305

import configparser  # for reading the parameters file
import sys  # for system errors and printouts
from pathlib import Path  # for paths of files
import os  # for reading the input data
import time  # for timing
import random #for shuffling
from tqdm import tqdm #for progress bar
import numpy as np


# Global parameters
parameter_file = 'default_parameters.ini'  # the main parameters file
data_main_directory = Path('data')  # the main path were all the data directories are
parameters_dictionary = dict()  # dictionary that holds the input parameters, key = parameter name, value = value
document_list = dict()  # dictionary of the input documents, key = document id, value = the document


# DO NOT CHANGE THIS METHOD
# Reads the parameters of the project from the parameter file 'file'
# and stores them to the parameter dictionary 'parameters_dictionary'
def read_parameters():
    config = configparser.ConfigParser()
    config.read(parameter_file)
    for section in config.sections():
        for key in config[section]:
            if key == 'data':
                parameters_dictionary[key] = config[section][key]
            elif key == 'naive':
                parameters_dictionary[key] = bool(config[section][key])
            elif key == 't':
                parameters_dictionary[key] = float(config[section][key])
            else:
                parameters_dictionary[key] = int(config[section][key])


# DO NOT CHANGE THIS METHOD
# Reads all the documents in the 'data_path' and stores them in the dictionary 'document_list'
def read_data(data_path):
    for (root, dirs, file) in os.walk(data_path):
        for f in file:
            file_path = data_path / f
            doc = open(file_path).read().strip().replace('\n', ' ')
            file_id = int(file_path.stem)
            document_list[file_id] = doc


# DO NOT CHANGE THIS METHOD
# Calculates the Jaccard Similarity between two documents represented as sets
def jaccard(doc1, doc2):
    return len(doc1.intersection(doc2)) / float(len(doc1.union(doc2)))


# DO NOT CHANGE THIS METHOD
# Define a function to map a 2D matrix coordinate into a 1D index.
def get_triangle_index(i, j, length):
    if i == j:  # that's an error.
        sys.stderr.write("Can't access triangle matrix with i == j")
        sys.exit(1)
    if j < i:  # just swap the values.
        temp = i
        i = j
        j = temp

    # Calculate the index within the triangular array. Taken from pg. 211 of:
    # http://infolab.stanford.edu/~ullman/mmds/ch6.pdf
    # adapted for a 0-based index.
    k = int(i * (length - (i + 1) / 2.0) + j - i) - 1

    return k


# DO NOT CHANGE THIS METHOD
# Calculates the similarities of all the combinations of documents and returns the similarity triangular matrix
def naive():
    comparisons=0
    docs_Sets = []  # holds the set of words of each document

    for doc in document_list.values():
        docs_Sets.append(set(doc.split()))

    # Using triangular array to store the similarities, avoiding half size and similarities of i==j
    num_elems = int(len(docs_Sets) * (len(docs_Sets) - 1) / 2)
    similarity_matrix = [0 for x in range(num_elems)]
    for i in tqdm(range(len(docs_Sets))):
        for j in range(i + 1, len(docs_Sets)):
            comparisons+=1
            similarity_matrix[get_triangle_index(i, j, len(docs_Sets))] = jaccard(docs_Sets[i], docs_Sets[j])
    print("Naive comparisons: ",comparisons)
    return similarity_matrix


# METHOD FOR TASK 1
# Creates the k-Shingles of each document and returns a list of them
def k_shingles_one_doc(document_file):
    k=parameters_dictionary["k"]
    with open(document_file,'r') as file:
        words=[]
        docs_k_shingles=[]# holds the k-shingles of each document

        for line in file:
            for word in line.lower().split():
                s = ''.join(c for c in word if c.isalnum())
                if s!="":
                    words.append(s)
        for i in range(len(words)):
            try:
                shingle=[]
                for j in range(k):
                    shingle.append(words[i+j])
                if not shingle in docs_k_shingles:
                    docs_k_shingles.append(shingle)
            except:
                pass
    return docs_k_shingles

def k_shingles():
    directory="data/"+parameters_dictionary["data"]
    total=[]
    for filename in tqdm(os.listdir(directory)):
        f = os.path.join(directory, filename)
        total.append(k_shingles_one_doc(f))
    return total
        
        
# METHOD FOR TASK 2
# Creates a signatures set of the documents from the k-shingles list
def signature_set(k_shingles):
    docs_sig_sets = []
    shingles = list(sorted(set(tuple(sub) for sub in sum(k_shingles, []))))
    size = len(k_shingles)
    k = parameters_dictionary["k"]

    sorted_docs = []
    for doc in k_shingles:
        sorted_docs.append(sorted(doc))
    k_shingles = sorted_docs

    for v in tqdm(shingles):
        temp_list = np.zeros(size)
        temp_list2 = np.array([i[0] if len(i) > 0 else np.array([""]*k) for i in k_shingles],dtype=object)
        ind = np.where((temp_list2 == np.array(v)).all(axis=1))
        for i in ind[0]:
            k_shingles[i].pop(0)
        temp_list[ind] = 1
        
        docs_sig_sets.append(temp_list)

    print(f"Number of shingles {len(shingles)}")

    return docs_sig_sets


# METHOD FOR TASK 3
# Creates the minHash signatures after simulation of permutations
def minHash(docs_signature_sets):
    pi=parameters_dictionary['permutations']
    permutation_matrix=[]
    docs_signature_sets = np.array(docs_signature_sets)
    doc_size = docs_signature_sets.shape[0]
    no_of_hashes = 50 #This is a value that we are setting. 
    tilfeldig=[]

    for j in range(1, (doc_size+1)//no_of_hashes):
        tilfeldig.append(j)
    tilfeldig = tilfeldig*no_of_hashes
    
    if len(tilfeldig) < doc_size:
        for i in range(doc_size-len(tilfeldig)):
            tilfeldig.append(random.randint(1, (doc_size+1)//no_of_hashes))
    for i in tqdm(range(pi)):
        random.shuffle(tilfeldig)      
        permutation_matrix.append(tilfeldig.copy())

    min_hash_signatures = []
    
    for i in tqdm(range(pi)):
        pi_iter=1
        signature_row=np.zeros(docs_signature_sets.shape[1]).tolist()
        while True:
            a=np.where(np.array(permutation_matrix[i]) == pi_iter)
            for j in range(len(docs_signature_sets[0])):#number of docs. Iterating through a doc.
                for k in a[0]:
                    if docs_signature_sets[k][j]==1:
                        if signature_row[j] ==0:
                            signature_row[j]=pi_iter
                            continue

            pi_iter+=1
            if ((0 not in signature_row) or (pi_iter>doc_size)):
                min_hash_signatures.append(signature_row)
                break
    return min_hash_signatures


# METHOD FOR TASK 4
# Hashes the MinHash Signature Matrix into buckets and find candidate similar documents
def lsh(m_matrix):
    no_of_buckets=parameters_dictionary["buckets"]
    r=parameters_dictionary["r"]
    candidates = []  # list of candidate sets of documents for checking similarity

    m_matrix = np.array(m_matrix)
    b = m_matrix.shape[0]//r
    start = 0
    end = start + r
    comparisons = 0
    for band in tqdm(range(b)):
        buckets = []
        bucket_candidates = []
        for column in range(m_matrix.shape[1]):
            temp = []
            try:
                for i in range(start, end):
                    temp.append(m_matrix[i][column])
                if not any(np.array_equal(x, temp) for x in buckets) and (len(buckets) < no_of_buckets):
                    buckets.append(temp)
                    bucket_candidates.append([column])
                else:
                    for index, bucket in enumerate(buckets):
                        comparisons += 1
                        if np.array_equal(temp, bucket):
                            bucket_candidates[index].append(column)
            except:
                pass
        for candidate in bucket_candidates:
            if len(candidate) > 1:
                candidates.append(candidate)
        start = end
        end = start + r
    for pair in candidates:
        if len(pair) > 2:
            for k in [(pair[i],pair[j]) for i in range(len(pair)) for j in range(i+1, len(pair))]:
                candidates.append(k)
            candidates = [i for i in candidates if i != pair]
    b_set = set(tuple(x) for x in candidates)
    candidates = [ list(x) for x in b_set ]
    print(f"number of comparisons = {comparisons}")
    return candidates



# METHOD FOR TASK 5
# Calculates the similarities of the candidate documents
def candidates_similarities(candidate_docs, min_hash_matrix):

    similarity_matrix=np.zeros(len(candidate_docs))

    for i in tqdm(range(len(candidate_docs))):
        nr1=candidate_docs[i][0]
        nr2=candidate_docs[i][1]

        for j in range(len(min_hash_matrix)):
            if min_hash_matrix[j][nr1]==min_hash_matrix[j][nr2]:
                similarity_matrix[i]+=1
        
    similarity_matrix/=len(min_hash_matrix)

    return similarity_matrix


# METHOD FOR TASK 6
# Returns the document pairs of over t% similarity
def return_results(lsh_similarity_matrix):
    t=parameters_dictionary['t']
    document_pairs = []
    count = 0
    for id, similarity in tqdm(enumerate(lsh_similarity_matrix)):
        if similarity > t:
            count += 1
            document_pairs.append(candidate_docs[id])
    print(f"There are {count} pairs.")

    return document_pairs


# METHOD FOR TASK 6
def count_false_neg_and_pos(lsh_similarity_matrix, naive_similarity_matrix):
    t = parameters_dictionary["t"]
    false_negatives = 0
    false_positives = 0
    total = 0
    for i in naive_similarity_matrix:
        if i > t:
            total += 1
    for id, similarity in enumerate(lsh_similarity_matrix):
        naive_sim = naive_similarity_matrix[get_triangle_index(candidate_docs[id][0], candidate_docs[id][1], len(document_list))]
        if similarity > t and naive_sim <= t:
            false_positives += 1
        elif similarity <= t and naive_sim > t:
            false_negatives += 1
    print(total)

    return false_negatives, false_positives


# DO NOT CHANGE THIS METHOD
# The main method where all code starts
if __name__ == '__main__':
    # Reading the parameters
    read_parameters()

    #Here, one can change the parameters.
    parameters_dictionary['naive']="true"

    # Reading the data
    print("Data reading...")
    data_folder = data_main_directory / parameters_dictionary['data']
    t0 = time.time()
    read_data(data_folder)
    document_list = {k: document_list[k] for k in sorted(document_list)}
    t1 = time.time()
    print(len(document_list), "documents were read in", t1 - t0, "sec\n")

    # Naive
    naive_similarity_matrix = []
    if parameters_dictionary['naive']:
        print("Starting to calculate the similarities of documents...")
        t2 = time.time()
        naive_similarity_matrix = naive()
        t3 = time.time()
        print("Calculating the similarities of", len(naive_similarity_matrix),
              "combinations of documents took", t3 - t2, "sec\n")

    # k-Shingles
    print("Starting to create all k-shingles of the documents...")
    t4 = time.time()
    all_docs_k_shingles = k_shingles()
    t5 = time.time()
    print("Representing documents with k-shingles took", t5 - t4, "sec\n")

    # signatures sets
    print("Starting to create the signatures of the documents...")
    t6 = time.time()
    signature_sets = signature_set(all_docs_k_shingles)
    t7 = time.time()
    print("Signatures representation took", t7 - t6, "sec\n")

    # Permutations
    print("Starting to simulate the MinHash Signature Matrix...")
    t8 = time.time()
    min_hash_signatures = minHash(signature_sets)
    t9 = time.time()
    print("Simulation of MinHash Signature Matrix took", t9 - t8, "sec\n")

    # LSH
    print("Starting the Locality-Sensitive Hashing...")
    t10 = time.time()
    candidate_docs = lsh(min_hash_signatures)
    t11 = time.time()
    print("LSH took", t11 - t10, "sec\n")

    # Candidate similarities
    print("Starting to calculate similarities of the candidate documents...")
    t12 = time.time()
    lsh_similarity_matrix = candidates_similarities(candidate_docs, min_hash_signatures)
    t13 = time.time()
    print("Candidate documents similarity calculation took", t13 - t12, "sec\n\n")

    # Return the over t similar pairs
    print("Starting to get the pairs of documents with over ", parameters_dictionary['t'], "% similarity...")
    t14 = time.time()
    pairs = return_results(lsh_similarity_matrix)
    t15 = time.time()
    print("The pairs of documents are:\n")

    # Count false negatives and positives
    if parameters_dictionary['naive']:
        print("Starting to calculate the false negatives and positives...")
        t16 = time.time()
        false_negatives, false_positives = count_false_neg_and_pos(lsh_similarity_matrix, naive_similarity_matrix)
        t17 = time.time()
        print("False negatives = ", false_negatives, "\nFalse positives = ", false_positives, "\n\n")

    if parameters_dictionary['naive']:
        print("Naive similarity calculation took", t3 - t2, "sec")

    print("LSH process took in total", t13 - t4, "sec")
    print("number of candidate_docs:",len(candidate_docs))

