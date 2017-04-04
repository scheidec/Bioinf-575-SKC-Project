#!/urs/env/bin python
"""Read in fasta to get the frequencies
"""

__author__='Nanxiang Zhao'
__email__='samzhao@umich.edu'


import feature_extraction as fe
import numpy as np


def ft_ext(fh):

    freq_list_lists=[]
    i = 0

    for line in fh:

        line = line.rstrip()
        i = i + 1

        if not i % 2 == 0:
            freq_list = []

        if line.startswith('>'):
            identifier = line.split()[0]
            freq_list.append(identifier)
        else:
            mono = fe.monofreq(line)
            for value in mono.values():
                freq_list.append(value)
            di = fe.difreq(line)
            for value in di.values():
                freq_list.append(value)

        if i % 2 ==0:
            freq_list_lists.append(freq_list)

    ary = np.array(freq_list_lists)
    return ary

#print ary[:,2]
