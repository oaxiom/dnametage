
import os
import math
import logging
import gzip
import numpy

from .miniglbase import genelist, glload, expression, draw
from . import preprocess
from .clock_data import clocks # coefficients are the values;

class methyl_age:
    def __init__(self):
        # -------------- set up the logger here.
        logging.basicConfig(level=logging.INFO,
                            #format='%(levelname)-8s: %(name)s : %(message)s', # Use this to trace orgue loggers
                            format='%(levelname)-8s: %(message)s',
                            datefmt='%m-%d %H:%M')
        
        # use config.log. ... () to get to the logger
        self.log = logging.getLogger('dnametage')
        
        _ = logging.getLogger('matplotlib').setLevel(logging.WARNING) # Bodge to silence the matplotlib logging
        _ = logging.getLogger('fontTools.subset').setLevel(logging.WARNING) # Another rogue logger
        
        self.cpgs = None # aka betas
        self.metadata = None
        self.sample_names = None
        self.ages = None
        self.clocks = clocks()
        self.draw = None
        self.epicv2 = None

    def load_cpgs_tsv(self, filename, gzipped=False, header=True, force_tsv=True):
        """
        Load the cpg files in the form:
        
        <identifier> <sample1> ... <samplen>
        
        """
        form = {'id': 0}
        
        if header:
            form.update({'skiplines': 1})
        
        self.cpgs = expression(filename=filename, format=form,
            expn='column[1:]', force_tsv=force_tsv, gzip=gzipped)

    def load_metadata(self, filename, gzipped=True, format=None, force_tsv=True):
        """
        Load the meta data.
        
        Must have two columns specified by the format specifier.
        
        sample_name (must match the cpg table condition names
        age
        
        other columns are optional.
        
        """
        assert self.cpgs, 'You must load the cpg table first'
        assert 'age' in format, 'format needs an age key'
        assert 'sample_name' in format, 'format needs a sample_name key'
        
        self.metadata = genelist(filename=filename, format=format, gzip=gzipped, force_tsv=force_tsv)
        
        # sanity checking on the metadata
        assert len(self.metadata) == len(self.cpgs.getConditionNames()), "the sample_names don't match the sizes of the cpg table"
        
        # convert the sample_names into a more useful order.
        self.map_sample_age = {}
        for data in self.metadata:
            self.map_sample_age[data['sample_name']] = data['age']
            
        # check the cpgs accessions
        if not self.cpgs['id'][0].startswith('cg'):
            self.log.warning('CpG sites do not start with cg, probably you will need to run remap_site_to_genome()')

    def __bsmap_rat_load(self, filename: str):
        form = {'loc': 'location(chr=column[0], left=int(column[1])+1, right=int(column[1])+1)',
                'ratio': 4}
        return delayedlist(filename=filename, format=form, force_tsv=True, gzip=True)

    def remap_site_to_genome(self, filename: str, input_format: str):
        """

        """
        assert input_format in ('bsmap'), f'{input_format} is not a valid input format, only bsmap has been implemented'

        self.log.error('Not implemented')
        return None

        if input_format == 'bsmap':
            data = self.__bsmap_rat_load(filename=filename)
            self.log.info('Loaded bsmap meth_ratio format file')

        if not self.epicv2:
            script_path = os.path.dirname(os.path.realpath(__file__))

            self.epicv2 = glload(filename=os.path.join(script_path, 'annotations/EPIVv2_annotations.glb'))
            self.log.info('Loaded EPICv2.hg38.manifest data')

        mapped = self.epicv2.map(genelist=data, key='loc')
        return mapped

    def run(self,  clock: str,
                   fit_meth: str = 'linear',
                   imputation: bool = True,
                   ):
        """
    
        :param betas:
        :param plot_filename:
        :param clock:
        :param fit_meth:
        :param imputation:
        :return:
        """
        assert self.cpgs, 'CpG data not found'
        assert clock in self.clocks, f'{clock} was not found in the valid clocks: {clocks.keys()}'
    
        # Preprocessing
        self.is_beta = True
        self.x_lim = [0, 100] # suggested sizes for scatters
        self.y_lim = [0, 100]
    
        if clock == 'Horvath_2013':
            self.log.info('Preprocess Horvath2013 clock')
            #cpgs = preprocess.horvath2013(self.cpgs, normalizeData=True)
    
        '''
        elif clock == 'ZhangQ2019':
            betas = preprocess_ZhangQ2019(betas)
    
        elif clock == 'YangZ2016':
            mAge < - EstEpiTOC(betas, epiTOCcpgs, mode="raw", ref.idx=NULL)
            m_age < - as.matrix(mAge)
            is_beta = FALSE
    
        elif clock == 'epiTOC2':
            m_age < - epiTOC2(betas, coefs, full_model=!simple_mode)
            m_age < - as.matrix(m_age)
            is_beta = FALSE
    
        elif clock == 'DunedinPACE':
            betas < - preprocessDunedinPACE(betas, ref_means=gold_standard_means)
            y_lim = c(0.5, 2)
    
        elif clock == 'BernabeuE2023c':
            coefs$Probe < - sub('_2', '', coefs$Probe_2)
    
        elif clock % in ('LuA2023p1', 'LuA2023p2', 'LuA2023p3'):
            if not MM_array:
                betas = arrayConverter_EPICtoMM(betas, coefs$Probe[-1])
                imputation = False
                log.warning('When clock == Lu clocks, imputation is set to False')
        '''
        """
        # Free the Y limits in plotting
        if clock in ('YangZ2016', 'ZhangY2017', 'LuA2019', 'FuentealbaM2025'):
            y_lim = None
        """
        """
        
        if self.is_beta:
            # This is the intersect
            r_coefs = coefs
            # data('HorvathS2013')
            coefs = setNames(coefs.Coefficient, coefs.Probe)
            ## add intercept
            betas = rbind(betas, intercept=1)
    
            ## identify missing probes, set their beta values as 0.5
            betas = betas[rownames(betas) in names(coefs), ]
            missing_probe = setdiff(names(coefs), rownames(betas))
            if len(missing_probe) > 0:
                log.warning("Found ", len(missing_probe), "out of", len(coefs), "probes missing! They will be assigned with mean values from reference dataset, missing probes are:\n ", missing_probe)
        """
        """
        if imputation:
            ## Mean imputation
            data(list='golden_ref', envir=environment())
            ref_mean = setNames(golden_ref.Mean, rownames(golden_ref))
            ref_mean = ref_mean[names(ref_mean) in names(coefs)]
            betas = mean_imputation(mt=betas, ref=ref_mean, only_ref_rows=FALSE)
        else:
            betas[betas != None] = 0
        """
    
        clock_data = self.clocks.get(clock)
        
        print(clock_data['coef'])
        
        # get only the matching ids;
        matched = clock_data['coef'].map(genelist=self.cpgs, key='id') # This will delete the intercept row

        # TODO Sanity checking for decent overlap %

        percent_clock_probes_matched = (len(matched) / len(clock_data['coef'])) * 100
        self.log.info(f'{percent_clock_probes_matched:.1f}% probes matched')
        if percent_clock_probes_matched < 90:
            self.log.warning('Less than 90% of probes matched!')
            self.log.warning('Suggest imputation of missing probes')

        # matrix multiplication
        cpg_array = matched.getExpressionTable()
        
        intercept = clock_data['coef'][0]['coef'] # intercept
        
        #print(cpg_array.T[0][0])
        #print(numpy.array(matched['coef'])[0])
        #print(intercept)
        m_age = (cpg_array.T * numpy.array(matched['coef']))   # matrix(data=clocks[rownames(betas)])

        #print(m_age[0][0])

        m_age = numpy.sum(m_age, axis=1) + intercept
    
        ## post transformation
        if clock in ('Horvath_2013', 'ShirebyG2020', 'HorvathS2018', 'McEwenL2019', 'PCHorvathS2013', 'PCHorvathS2018'):
            def anti_trafo(x, adult_age=20):
                if x < 0:
                    return (1+adult_age) * math.exp(x) - 1
                else:
                    return (1+adult_age) * x + adult_age
    
            predAge = [anti_trafo(x) for x in m_age]
    
        '''
        elif clock in ('CBL_specific', 'CBL_common', 'Cortex_common'):
            m_age[, 1] = math.exp(m_age[, 1])
    
        elif clock % in %c('BernabeuE2023c'):
            ## log predictor for samples younger than 20
            less_20 < - m_age[, 1] < 20
            if sum(less_20) > 0:
                m_age_L = t(betas[less_20, ]) * matrix(data=coefs_L[rownames(m_age)[less_20]])
                m_age[less_20, 1] = math.exp(m_age_L[, 1])
    
        elif clock in ('LuA2023p1', ):
            m_age[, 1] < - exp(m_age[, 1]) - 2
    
        elif clock in ('LuA2023p2',):
            data(list='SpeciesAgeInfo', envir=environment())
            my_max = ifelse(species % in %c('Homo sapiens', 'Mus musculus'), 1, 1.3)
            y_maxAge = SpeciesAgeInfo[species, 'maxAge'] * my_max
            y_gestation = SpeciesAgeInfo[species, 'GestationTimeInYears']
            m_age[, 1] = exp(-exp(-1 * m_age[, 1]))
            m_age[, 1] = m_age[, 1] * (y_maxAge + y_gestation) - y_gestation
    
        elif clock in ('LuA2023p3'):
            data(list='SpeciesAgeInfo', envir=environment())
            y_gestation < - SpeciesAgeInfo[species, 'GestationTimeInYears']
            y_ASM < - SpeciesAgeInfo[species, 'averagedMaturity.yrs']
            F2_revtrsf_clock3 < - function(y.pred, m1){
                ifelse(y.pred < 0, (exp(y.pred)-1) * m1 + m1, y.pred * m1+m1)  # wyc
                }
            a_Logli < - 5 * (y_gestation / y_ASM) ^ 0.38
            r_adult_age < - F2_revtrsf_clock3(m_age[, 1], a_Logli)
            m_age[, 1] < - r_adult_age * (y_ASM + y_gestation) - y_gestation
    
        '''
        
        ## save results
        self.predicted_ages = m_age # data.frame(Sample=m_age, mAge=m_age)
        
        # get a table;
        load_list = [{'id': id, 'predage': predage, 'actual_age': age} for id, predage, age in zip(matched.getConditionNames(), predAge, self.metadata['age'])]
    
        gl = genelist()
        gl.load_list(load_list)
        
        self.final_data = gl
        
        return gl
        
    def scatter(self, filename):
        """
        Draw a scatter of pred age versus actual age;
        
        """
        assert self.final_data, 'run() has not completed'
    
        if not self.draw:
            self.draw = draw.draw()
            
        fig = self.draw.getfigure()
        ax = fig.add_subplot(111)
        
        x = self.final_data['predage']
        y = self.final_data['actual_age']
        
        ax.scatter(x, y)
        
        ax.set_xlim([0,100])
        ax.set_ylim([0,100])
        
        self.draw.savefigure(fig, filename)
        
        
        
