from pyserini.search.lucene import LuceneSearcher
import sys

searcher = LuceneSearcher(sys.argv[1])
with open(sys.argv[2],'r') as f:
    for l in f:
        l = l.strip().split('\t')
        hits = searcher.search(l[0], 1000)
        l = '\t'.join(l)
        for i in range(len(hits)):
            print("{}\t{}\t{}\t{}".format(l,i+1, hits[i].docid, hits[i].score))
