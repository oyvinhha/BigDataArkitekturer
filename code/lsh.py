# This is the code for the LSH project of TDT4305

import configparser  # for reading the parameters file
import sys  # for system errors and printouts
from pathlib import Path  # for paths of files
import os  # for reading the input data
import time  # for timing
import random
from tqdm import tqdm
import numpy as np
import math
 


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
    docs_Sets = []  # holds the set of words of each document

    for doc in document_list.values():
        docs_Sets.append(set(doc.split()))

    # Using triangular array to store the similarities, avoiding half size and similarities of i==j
    num_elems = int(len(docs_Sets) * (len(docs_Sets) - 1) / 2)
    similarity_matrix = [0 for x in range(num_elems)]
    for i in tqdm(range(len(docs_Sets))):
        for j in range(i + 1, len(docs_Sets)):
            similarity_matrix[get_triangle_index(i, j, len(docs_Sets))] = jaccard(docs_Sets[i], docs_Sets[j])

    return similarity_matrix


# METHOD FOR TASK 1
# Creates the k-Shingles of each document and returns a list of them
def k_shingles_one_doc(document_file):
    k=parameters_dictionary["k"]

    with open(document_file,'r') as file:
        words=[]
        docs_k_shingles=[]# holds the k-shingles of each document

        for line in file:
            for word in line.split():
                words.append(word)
    
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
        # checking if it is a file
        #if os.path.isfile(f):
            #print(f)
        total.append(k_shingles_one_doc(f))
    #print("total",total)
    return total
        
        
# METHOD FOR TASK 2
# Creates a signatures set of the documents from the k-shingles list
def signature_set(k_shingles):
    docs_sig_sets = []

    # implement your code here
    #print(list(document_list.keys())[-1])
    #signature = []

    shingles = [[el] for el in np.unique(np.array(k_shingles).flatten()).tolist()]

    print("Total amount of shingles: ",len(shingles))
    for i, v in tqdm(enumerate(shingles)):
        temp_list = np.zeros(len(k_shingles))
        for ind, document in enumerate(k_shingles):
            if v in document:
                temp_list[ind] = 1
        docs_sig_sets.append(list(temp_list))
        #print(temp_list)
    #print(docs_sig_sets)
    print(f"Number of shingles {len(shingles)}")
    return docs_sig_sets


# METHOD FOR TASK 3
# Creates the minHash signatures after simulation of permutations
def minHash(docs_signature_sets):
    pi=parameters_dictionary['permutations']
    #print("docs_signature_sets:",docs_signature_sets)
    permutation_matrix=[]
    for i in tqdm(range(pi)):
        tilfeldig=[]
        for j in range(len(docs_signature_sets)):           #shape[1]??
            tilfeldig.append(j)
        random.shuffle(tilfeldig)
        permutation_matrix.append(tilfeldig)
    #print("permutation matrix:",permutation_matrix)

    min_hash_signatures = []
    
    for i in tqdm(range(pi)):
        pi_iter=0
        signature_row=np.empty(len(docs_signature_sets[0]))
        signature_row=signature_row.tolist()
        #for iter in range(len(docs_signature_sets)):
        while True:
            #print("sigrow of 0's",signature_row)#ok
            a=permutation_matrix[i].index(pi_iter)          #a=rad 7,5,1,...
            for j in range(len(docs_signature_sets[0])):#number of docs. Iterating through a doc.
                if docs_signature_sets[a][j]==1:
                    #print("sigrow",signature_row)
                    if signature_row[j] ==0:
                        signature_row[j]=pi_iter
                        #print("sigrow",signature_row)

            pi_iter+=1
            #print(signature_row)
            #print("docs_signature_sets",docs_signature_sets)
            #print("len",len(docs_signature_sets))
            if ((not(0 in signature_row)) or (pi_iter>=len(docs_signature_sets))):
                #print("pi_iter",pi_iter)
                min_hash_signatures.append(signature_row)
                break
    

    #print("min_hash_signatures:",min_hash_signatures)
    return min_hash_signatures


# METHOD FOR TASK 4
# Hashes the MinHash Signature Matrix into buckets and find candidate similar documents
def lsh(m_matrix):
    no_of_buckets=parameters_dictionary["buckets"]
    r=parameters_dictionary["r"]
    candidates = []  # list of candidate sets of documents for checking similarity

    # implement your code here
    b = len(m_matrix)//r
    start = 0
    end = start + r
    comparisons = 0
    for band in tqdm(range(b)):
        buckets = [[] for _ in range(no_of_buckets)]
        bucket_candidates = [[] for _ in range(no_of_buckets)]
        for row in range(len(m_matrix[0])):
            temp = []
            try:
                for i in range(start, end):
                    comparisons += 1
                    temp.append(m_matrix[i][row])
                #print(temp)
                #print(buckets)
                for index, bucket in enumerate(buckets):
                    if temp not in buckets:
                        buckets[row] = temp
                        bucket_candidates[row].append(row)
                        break
                    else:
                        if temp == bucket:
                            bucket_candidates[index].append(row)
            except:
                pass
        #print("v",bucket_candidates)
        for candidate in bucket_candidates:
            if len(candidate) > 1:
                candidates.append(candidate)
        start = end
        end = start + r
    print(candidates)
    for pair in candidates:
        print(pair, len(pair))
        if len(pair) > 2:
            for k in [(pair[i],pair[j]) for i in range(len(pair)) for j in range(i+1, len(pair))]:
                candidates.append(k)
            candidates = [i for i in candidates if i != pair]
    #print(candidates)
    b_set = set(tuple(x) for x in candidates)
    candidates = [ list(x) for x in b_set ]
    print(f"number of comparisons = {comparisons}")
    return candidates



# METHOD FOR TASK 5
# Calculates the similarities of the candidate documents
def candidates_similarities(candidate_docs, min_hash_matrix):
    """For the candidate document pairs from the previous task, calculate the document
signature sets similarity using the fraction of the hash functions which they agree,
i.e.
similarity(d1, d2) = #(hi(d1) == hi(d2))
permutations"""
    #candidate_docs [[4, 5], [2, 4]]
    #print("lengde på candidates",len(candidate_docs))
    #min_hash_matrix [[5, 7, 3, 1, 1, 1.0], [5, 5, 5, 2, 1, 1.0], [2, 3, 4, 1, 4, 1.0], [1, 1, 1, 2, 1, 1.0]]

    similarity_matrix=np.zeros(len(candidate_docs))

    for i in tqdm(range(len(candidate_docs))):
        nr1=candidate_docs[i][0]
        nr2=candidate_docs[i][1]

        for j in range(len(min_hash_matrix)):
            if min_hash_matrix[j][nr1]==min_hash_matrix[j][nr2]:
                similarity_matrix[i]+=1
        
    similarity_matrix/=len(min_hash_matrix)

    #print("sim_matrix:",similarity_matrix)
    # implement your code here

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
    # implement your code here

    return document_pairs


# METHOD FOR TASK 6
def count_false_neg_and_pos(lsh_similarity_matrix, naive_similarity_matrix):
    t = parameters_dictionary["t"]
    #print(naive_similarity_matrix[:10])
    false_negatives = 0
    false_positives = 0
    for id, similarity in enumerate(lsh_similarity_matrix):
        naive_sim = naive_similarity_matrix[get_triangle_index(candidate_docs[id][0], candidate_docs[id][1], len(document_list))]
        #print(naive_sim)
        if similarity > t and naive_sim <= t:
            false_positives += 1
        elif similarity <= t and naive_sim > t:
            false_negatives += 1


    # implement your code here

    return false_negatives, false_positives


# DO NOT CHANGE THIS METHOD
# The main method where all code starts
if __name__ == '__main__':
    
    # Reading the parameters
    read_parameters()
    parameters_dictionary['data']="test"                            #GOING THROUGH THE TEST DATA
    parameters_dictionary['naive']="true"
    parameters_dictionary['k']=1

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
    all_docs_k_shingles = k_shingles()#k_shingles('data/test')
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
    print("LSH candidate docs: ",candidate_docs)

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
    #for p in pairs:
    #    print(p)
    #print("\n")

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
    #print(parameters_dictionary)
    print("candidate_docs:",len(candidate_docs))

