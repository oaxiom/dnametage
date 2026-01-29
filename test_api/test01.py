
import sys
sys.path.append('../')

import dnametagelib

dnamet = dnametagelib.methyl_age()

dnamet.load_cpgs_tsv('../test_data/Horvath2013/gb-2013-14-10-r115-S26.csv.gz', force_tsv=False, gzipped=True)
dnamet.load_metadata('../test_data/Horvath2013/gb-2013-14-10-r115-S27.csv.gz', force_tsv=False, format={'sample_name': 1, 'age':7})

dnamet.run('Horvath_2013', )