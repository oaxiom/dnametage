# extract the cg number to location tables.

import sys
sys.path.insert(0, '../')

import miniglbase

form = {'force_tsv': True, 'loc': 'location(chr=column[0], left=column[1], right=column[2])', 'id': 8, 'skiplines': 1, 'debug': 10}

gl = miniglbase.genelist(filename='EPICv2.hg38.manifest.tsv.gz', format=form, gzip=True)

print(gl)