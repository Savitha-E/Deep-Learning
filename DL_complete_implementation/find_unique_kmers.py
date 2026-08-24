seq1 = "GTTACTTATGCCCGATGACGGATGCTAGGGCTAGATGATCGATAT"


def find_unique_kmers(seq, k):
    kmers_list = []
    for i in range(len(seq)):
        kmer = seq[i:k]
        k += 1
        if kmer not in kmers_list:
            kmers_list.append(kmer)

    return kmers_list[0:len(kmers_list) - 3]


kmers = find_unique_kmers(seq1, 4)
print(kmers)
