dog <- function(verbose=T, ...){
	if (verbose) cat("[init]", ..., "\n")
}

#kullback-leibler divergence of two negative multinomials with marginal means mus1 and mus2
#r is the same for both distributions
#that's a non-symmetric distance, it's the distance from mus1 to mus2
#vectorized on the second argument
kldiv <- function(mus1, mus2, r=Inf){
	if (!is.matrix(mus2)) dim(mus2) <- c(length(mus2), 1)
	ratioTerm <- mus1*log(mus1/mus2)
	ratioTerm[mus1==0] <- 0
	ratioTerm[mus1>0&mus2==0] <- Inf
	if (r==Inf){
		return(colSums(mus2-mus1+ratioTerm))
	} else {
		mu1 <- sum(mus1)
		mu2 <- colSums(mus2)
		return((mu1+r)*log((mu2+r)/(mu1+r)) + colSums(ratioTerm))
	}
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
	mycov <- rowcov(dccounts, besselCorr=besselCorr, nthreads=nthreads)
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
	sdev <- sqrt(edc$values)
	rotation <- edc$vectors
	cn <- paste0("PC", 1L:ncol(mycov))
	colnames(rotation) <- cn
	rownames(rotation) <- rownames(counts)
	#here again... memory allocation
	#is the bottleneck and it is much faster in R...
	scores <- t(rotation) %*% dccounts
	
	list(sdev=sdev, cov=mycov, rotation=rotation, tx=scores, center=ctr, scale=d)
}

#pprior makes everything as if when we observe column i, we observe column i 1-pprior times and pprior times another random column
initCool <- function(counts, k, nlev=5, nthreads=1, nbtype=c("indep", "dep", "pois"), axes=c("counts", "pca"), pprior=0.01, verbose=F){
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
		dog("splitting axes (involves sorting)", v=verbose)
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
	dmat <- apply(mus, 2, kldiv, mus2=mus, r=r)
	#symmetrized kl divergence
	dmat <- (dmat + t(dmat))/2
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
	posteriors <- matrix(0, nrow=k, ncol=ncol(counts)) 
	fillPosteriors(coords, multidclust, posteriors, nthreads)
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
