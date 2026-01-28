
import random, math

def imputation(dat1,
               goldstandard=None, # probeAnnotation21kdatMethUsed.goldstandard2,
               fastImputation: bool = False,
               log = None
               ):
    """

    :param dat1:
    :param goldstandard:
    :param fastImputation:
    :param log:
    :return:
    """

    # STEP 3: Create the output file called datout
    random.seed(1)
    
    datMethUsed = t(dat1[:, -1])
    colnames.datMethUsed = as_character(dat1[:, 1])
    noMissingPerSample = rowSums(is_na(datMethUsed))
    max_noMissingPerSample = max(noMissingPerSample, na_rm = True)

    log.info("Start data imputation ...")
    # STEP 2: Imputing
    if (not fastImputation & nrow(datMethUsed) > 1 and max_noMissingPerSample < 3000 ):
        # run the following code if there is at least one missing
        if (max_noMissingPerSample > 0):
            if (not requireNamespace("impute", quietly=True)):
                BiocManager.install("impute")
                require("impute")
            else:
                require("impute")

            dimnames1 = dimnames(datMethUsed)
            datMethUsed = data.frame(t(impute.knn(t(datMethUsed), maxp=ncol(datMethUsed)).data))
            dimnames.datMethUsed = dimnames1


def horvath2013(log):
    """
    Preprocess the Horvath clock data
    :return:

    """
    ### ORIGINAL AUTHOR: Steve Horvath
    ### Adopted by Yucheng Wang
    ### modifications: only keep some ensential codes
    ### Python port Andrew Hutchins

    if (max(noMissingPerSample, na_rm=TRUE) >= 3000): 
        fastImputation=True

    if (fastImputation or nrow(datMethUsed) == 1):
        if (max_noMissingPerSample >= 3000):
            normalizeData = False
        
        # run the following code if there is at least one missing
        if (max_noMissingPerSample > 0 and max_noMissingPerSample < 3000):
            dimnames1 < - dimnames(datMethUsed)
            for i in which(noMissingPerSample > 0):
                selectMissing1 = is_na(datMethUsed[i,:])
                datMethUsed[i, selectMissing1] = as_numeric(goldstandard[selectMissing1])
    
            dimnames.datMethUsed = dimnames1
    
        log.info("Finished imputation.")
        return (datMethUsed)
    

def horvathPreprocess(betas, normalizeData: bool = True, log=False):
    # load('horvath_clock.RData')  ## datClock, probeAnnotation21kdatMethUsed
    # probeAnnotation21kdatMethUsed <- read.table('../coefs/27k_reference.txt', header=TRUE)
    data("27k_reference", envir=environment())
    
    # STEP 2: Restrict the data to 21k probes and ensure they are numeric
    match1 = match(probeAnnotation21kdatMethUsed.Name, rownames(betas))
    if (sum(is_na(match1)) > 0):
        log.warning("CpG probes cannot be matched in horvath's ref probes, will set to NA")
    
    betas = merge(probeAnnotation21kdatMethUsed, betas, sort=False, by_x = 'Name', by_y = "row.names", all_x = TRUE)[:,-2]
    betas = imputation(betas, probeAnnotation21kdatMethUsed.goldstandard2)
    
    # STEP 3: Data normalization (each sample requires about 8 seconds). It would be straightforward to parallelize this operation.
    if normalizeData:
        log.info("Normalization by adusted BMIQ ...")
        log.info("Estimate running time for normalisation (BMIQ with fixed reference) by a single core: ", round(7.4 * nrow(betas) / 60, 1), " minutes.")
        betas = BMIQcalibration(datM=betas, goldstandard_beta=probeAnnotation21kdatMethUsed.goldstandard2, plots=False, use_cores=1)

    return betas
