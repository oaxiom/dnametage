
import math
import logging
import gzip

from . import miniglbase
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

    def load_cpgs_tsv(self, filename, gzipped=False, header=True, force_tsv=True):
        """
        Load the cpg files in the form:
        
        <identifier> <sample1> ... <samplen>
        
        """
        form = {'id': 0}
        
        if header:
            form.update({'skiplines': 1})
        
        self.cpgs = miniglbase.expression(filename=filename, format=form, 
            expn='column[1:]', force_tsv=force_tsv, gzip=gzipped)
        
        print(self.cpgs)

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
        
        self.metadata = miniglbase.genelist(filename=filename, format=format, gzip=gzipped, force_tsv=force_tsv)
        
        # sanity checking on the metadata
        print(self.metadata)

    def setup(self, cpgs,
                   plot_filename: str,
                   clock: str,
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
        assert clock in clocks, f'{clock} was not found in the valid clocks: {clocks.keys()}'
    
        # Preprocessing
        is_beta = True
        x_lim = [0, 100]
        y_lim = [0, 100]
    
        if clock == 'Horvath_2013':
            self.log.info('Preprocess Horvath2013 clock')
            cpgs = preprocess.horvath2013(cpgs, normalizeData=True)
    
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
    
        # Free the Y limits in plotting
        if clock in ('YangZ2016', 'ZhangY2017', 'LuA2019', 'FuentealbaM2025'):
            y_lim = None
    
        if is_beta:
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
    
        if imputation:
            ## Mean imputation
            data(list='golden_ref', envir=environment())
            ref_mean = setNames(golden_ref.Mean, rownames(golden_ref))
            ref_mean = ref_mean[names(ref_mean) in names(coefs)]
            betas = mean_imputation(mt=betas, ref=ref_mean, only_ref_rows=FALSE)
        else:
            betas[betas != None] = 0
    
        # matrix multiplication
        m_age = t(betas) * matrix(data=clocks[rownames(betas)])
    
        ## post transformation
        if clock in ('HorvathS2013', 'ShirebyG2020', 'HorvathS2018', 'McEwenL2019', 'PCHorvathS2013', 'PCHorvathS2018'):
    
            def Horvath2013_transform(x):
                if x > 0:
                    x = x * (20 + 1) + 20
                else:
                    x = math.exp(x + math.log(20 + 1)) - 1
    
            m_age[:, 1] = sapply(m_age[:, 1], HorvathS2013_transform)
    
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
    
        ## calculate age acceleration
        """
        warning_message = "\n'age_info' should be a dataframe which contains sample ID and age information, like:\nSample\tAge\nname1\t30\nname2\t60\nname3\t40\nAge acceleration will not be calculated."
        if insintance(age_info, list):
    
        if (all(c('Sample', 'Age') % in % colnames(age_info))){
            m_age < - merge(age_info, m_age, by='Sample', sort=FALSE)
            if (nrow(m_age) < 1){
                stop(message("Colnames of the input beta dataframe do not match any of the values of the 'Sample' column in age_info!"))
            }
            if (clock == 'PCGrimAge'){
                if ('Sex' % in %colnames(age_info)){
                    # m_age$is_Female <- NA
                    m_age$is_Female < - gsub('^F.*', 1, m_age$Sex)
                    m_age$is_Female < - gsub('^M.*', 0, m_age$is_Female)
                    m_age$is_Female < - as.numeric(m_age$is_Female)
    
                    m_age$mAge < - m_age$mAge + as.matrix(m_age[, c('is_Female', 'Age')]) % * %PCGrimAge_agesex$PCGrimAge
                } else {
                stop(message("\nTo calculate 'PCGrimage', 'age_info' should include a 'Sex' column that contains binary sex annotation, i.e. either Female or Male."))
                }
            }
    
            if ("Color" % in %colnames(age_info)){
                point_color < - m_age$Color
            } else {
                point_color < - NA
            }
    
            if ("Shape" % in %colnames(age_info)){
                point_pch < - m_age$Shape
            } else {
                point_pch < - NA
            }
    
            m_age$Age_Acceleration < - NA
            m_age$Age_Acceleration[! is.na(m_age$Age)] < - getAccel(m_age$Age[! is.na(m_age$Age)], m_age$mAge[! is.na(m_age$Age)], , method=fit_method, title=clock, do_plot=do_plot, point_color=point_color, point_shape=point_pch, simple=plot_simple, x_lim=x_lim, y_lim=y_lim)
            } else {
            warning(message("\nThe colnames of age_info should include both 'Sample' and 'Age', like:\nSample\tAge\nname1\t30\nname2\t60\nname3\t40\nAge\nAge acceleration will not be calculated."))
            }
    
            } else if (is.na(age_info[1])){
            if (clock == 'PCGrimAge'){
            stop(message("\nTo calculate 'PCGrimage': \n'age_info' should be a dataframe which contains sample ID, age, sex information, like:\nSample\tAge\tSex\nname1\t30\tFemale\nname2\t60\tMale\nname3\t40\tFemale\n"))
            }
    
         } else {
            warning(message(warning_message))
        }
        """
        ## save results
        m_age = data.frame(Sample=m_age, mAge=m_age)
    
        return m_age
    
