
import os
from . import miniglbase
import numpy

clock_files = {
    #https://static-content.springer.com/esm/art%3A10.1186%2Fgb-2013-14-10-r115/MediaObjects/13059_2013_3156_MOESM3_ESM.csv # coefficients;
    #https://static-content.springer.com/esm/art%3A10.1186%2Fgb-2013-14-10-r115/MediaObjects/13059_2013_3156_MOESM22_ESM.csv # reference featurs
    'Horvath_2013': {
        'coef': {
            'filename': "data/13059_2013_3156_MOESM3_ESM.csv.gz",
            'form': {"id": 0, "coef": 1, "skiplines": 2},
            'intercept': None,
            'postprocess': None,
            },
        'goldstandard': {
            'filename': "data/13059_2013_3156_MOESM22_ESM.csv.gz",
            'form': {"id": 0, "goldstandard": 6},
            'intercept': None,
            'postprocess': None,
            },
        },
    
    'GrimAge2': {
        'coef': None,
        'goldstandard': None
    },

}

class clocks: 
    def __init__(self):
        self.valid_clocks = clock_files.keys()
        self.loaded_data = {}
    
    def get(self, clock: str):
        assert clock in self.valid_clocks, f'{clock} not found'
        
        if clock in self.loaded_data:
            return self.loaded_data[clock]
        
        force_tsv = True
        if '.csv' in clock_files[clock]['coef']['filename']:
            force_tsv = False
            
        gzipped = False
        if clock_files[clock]['coef']['filename'].endswith('.gz'):
            gzipped = True
            
        script_path = os.path.dirname(os.path.realpath(__file__))
        # load data
        self.loaded_data[clock] = {
            'coef': miniglbase.genelist(filename=os.path.join(script_path, clock_files[clock]['coef']['filename']), format=clock_files[clock]['coef']['form'], force_tsv=force_tsv, gzip=gzipped),
            'goldstandard': miniglbase.genelist(filename=os.path.join(script_path, clock_files[clock]['goldstandard']['filename']), format=clock_files[clock]['goldstandard']['form'], force_tsv=force_tsv, gzip=gzipped),
            }
            
        return self.loaded_data[clock]
    
    def __contains__(self, clock):
        return clock in self.valid_clocks
        
        


