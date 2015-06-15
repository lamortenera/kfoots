dog <- function(verbose=T, ...){
    if (verbose) cat("[init]", ..., "\n")
}

tpca <- function(counts, center=T, scale=F, besselCorr=T, nthreads=1){
    #only the covariance computation is parallelized, unfortunately
    if (center) {
        #there was no way of parallelizing this... 
        #memory allocation here is the bottleneck and it is much faster in R...
        ctr <- rowMeans(counts)
        dccounts <- counts - ctr
    } else {
        ctr <- F
        dccounts <- counts; storage.mode(dccounts) <- "double"
    }
    mycov <- rowdotprod(dccounts, besselCorr=besselCorr, nthreads=nthreads)
    rownames(mycov) <- rownames(counts)
    colnames(mycov) <- rownames(counts)
    if (scale) {
        d <- sqrt(diag(mycov))
        mycov <- mycov/outer(d,d)
        dccounts <- dccounts/d
    } else {
        d <- F
    }
    
    edc <- eigen(mycov, symm=T)
    #force eigenvalues to be all positive (or null)
    edc$values[edc$values < 0] <- 0
    sdev <- sqrt(edc$values)
    rotation <- edc$vectors
    #define the sign of the vectors in a robust way, so that
    #small changes in the input will not flip the vectors
    vsign <- apply(rotation, 2, function(col) sign(col[which.max(abs(col))]))
    rotation <- rotation*vsign[col(rotation)]
    cn <- paste0("PC", 1L:ncol(mycov))
    colnames(rotation) <- cn
    rownames(rotation) <- rownames(counts)
    #here again... memory allocation
    #is the bottleneck and it is much faster in R...
    scores <- t(rotation) %*% dccounts
    
    list(sdev=sdev, cov=mycov, rotation=rotation, tx=scores, center=ctr, scale=d)
}

#pprior makes everything as if when we observe column i, we observe column i 1-pprior times and pprior times another random column
initAlgo <- function(counts, k, nlev=5, nthreads=1, nbtype=c("indep", "dep", "pois"), axes=c("counts", "pca"), pprior=0.01, verbose=F){
    axes <- match.arg(axes)
    nbtype <- match.arg(nbtype)
    #nlev says how many cuts we should do per axis
    #too many cuts will create seeds that are outliers
    #too little cuts will not explore the space sufficiently
    #we choose a default nlev of 5, making sure that that's enough to get k clusters
    nlev <- max(nlev,ceiling(k/nrow(counts)))
    nmark <- nrow(counts)
    priormus <- rowMeans(counts)
    if (axes == "counts"){
        dog("splitting axes", v=verbose)
        pca <- NULL
        coords <- splitAxesInt(counts, nlev, nthreads=nthreads)
    } else {
        dog("performing PCA on the count matrix", v=verbose)
        pca <- tpca(counts, nthreads=nthreads)
        #scores has the same format as counts
        scores <- pca$tx
        #linearly transforms the scores to positive integers from 0 to 999
        dog("splitting axes", v=verbose)
        coords <- splitAxes(scores, nlev, nthreads=nthreads)
    }
    
    onedclust <- lapply(1:nrow(counts), function(i) 0:(nlev-1))
    dog("computing average seeds", v=verbose)
    onedcenters <- clusterAverages2(counts, coords, onedclust, nthreads)
    mus <- onedcenters$mus*(1-pprior) + pprior*priormus
    sizes <- onedcenters$sizes
    member <- cbind(rep(1:nrow(counts), each=nlev), rep(1:nlev, nrow(counts))); colnames(member) <- c("mark", "level")
    rownames(mus) <- rownames(counts)
    colnames(mus) <- paste(rownames(coords)[member[,1]], member[,2], sep=":")
    
    if (sum(sizes>0) < k) stop("unable to find enough seeds, try to reduce the number of clusters")
    #discard empty clusters
    mus <- mus[, sizes>0]
    member <- member[sizes>0,]
    sizes <- sizes[sizes>0]
    
    #get a suitable r for the kullback-leibler divergence (even though it doesn't change much...)
    #and for the final models
    dog("estimating r parameter", v=verbose)
    r <- Inf
    ucs <- mapToUnique(colSumsInt(counts, nthreads))
    if (nbtype=="indep" || nbtype=="dep") r <- fitNB(ucs)$r
    dog("computing KL divergences", v=verbose)
    dmat <- KL_dist_mat(mus, r, nthreads=nthreads)
    #do hierarchical clustering
    dog("hierarchical clustering", v=verbose)
    hc <- hclust(as.dist(dmat), method="average", members=sizes)
    clust <- cutree(hc, k)
    #make partition matrix
    partition <- matrix(-1, nrow=nmark, ncol=nlev); rownames(partition) <- rownames(coords);
    partition[member] <- clust
    #collapse the cluster assignments
    multidclust <- lapply(seq_along(onedclust), function(mark) {partition[mark, ][onedclust[[mark]] + 1]-1})
    csizes <- sumAt(sizes, clust, k, F)
    #prepare posterior matrix
    dog("filling in posterior matrix", v=verbose)
    posteriors <- fillPosteriors(coords, multidclust, k, nthreads)
    priorcol <- csizes/ncol(counts)
    posteriors <- (1-pprior)*posteriors + pprior*priorcol
    
    #fit models
    #dummy variable, not needed...
    old_models <- list()
    for (i in 1:k){ old_models[[i]] <- list(mu=-1, r=-1, ps=numeric(nrow(counts))) }
    dog("fitting models", v=verbose)
    models <- fitModels(counts, posteriors, old_models, ucs=ucs, type=nbtype, nthreads=nthreads)
    for (i in seq_along(models)) names(models[[i]]$ps) <- rownames(counts)
    mix_coeff <- csizes/sum(csizes)
    
    list(models=models, mix_coeff=mix_coeff)
}


#to be parallelized, or avoid iteration through the whole matrix
rndModels <- function(counts, k, bgr_prior=0.5, ucs=NULL, nbtype="dep", nthreads=1){
    seeds <- getUniqueSeeds(counts, k)
    modelsFromSeeds(counts, seeds, bgr_prior=bgr_prior, ucs=ucs, nbtype=nbtype, nthreads=nthreads)
}

#seeds are columns of the count matrix which are guaranteed to be distinct.
#they are used to initialize the models
modelsFromSeeds <- function(counts, seeds, bgr_prior=0.5, ucs=NULL, nbtype="dep", nthreads=1){
    posteriors <- matrix(nrow=length(seeds), ncol=ncol(counts), bgr_prior)
    #perturb row i at column seeds[i]
    perturb_pos <- seeds*length(seeds) + 1:length(seeds)
    posteriors[perturb_pos] <- 1 + bgr_prior
    
    #initialize empty models
    models <- list()
    for (i in seq_along(seeds)){ models[[i]] <- list(mu=-1, r=-1, ps=numeric(nrow(counts))) }
    
    fitModels(counts, posteriors, models, ucs=ucs, type=nbtype, nthreads=nthreads)
}

#find k unique columns in the count matrix
getUniqueSeeds <- function(counts, k){
    shuffle <- sample(ncol(counts), ncol(counts))
    findUniqueSeeds(counts, shuffle, k)
}
