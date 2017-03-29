#!/urs/env/bin python
""" Feature Extraction
"""

__author__ = 'Nanxiang Zhao'
__email__ = 'samzhao@umich.edu'


#rna = 'GATCATCGGATGCTTAGGGGGATCGATCGGAGGCATT'

def monofreq(seq):

    all_counts = {}
    for base in ['A', 'T', 'G', 'C']:
        count = seq.count(base)
        all_counts[base] = count
    for key in all_counts:
        all_counts[key] = round(float(all_counts[key]) / len(seq),5)
    return(all_counts)

#print monofreq(rna)

def difreq(seq):

    all_counts = {}
    for base1 in ['A', 'T', 'G', 'C']:
        for base2 in ['A', 'T', 'G', 'C']:
            dinucleotide = base1 + base2
            count = seq.count(dinucleotide)
            all_counts[dinucleotide] = count
    for key in all_counts:
        length = len(seq) - 1
        all_counts[key] = round(float(all_counts[key]) / length,5)
    return(all_counts)

#print difreq(rna)
