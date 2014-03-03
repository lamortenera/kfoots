#for some weird reason the R matrix constructor is extremely slow
createZeroMatrix <- function(nrow, ncol){
	v <- numeric(nrow*ncol)
	dim(v) <- c(nrow, ncol)
	v
}

#twice slower as the zero matrix
createMatrix <- function(value, nrow, ncol){
	v <- rep(value, nrow*ncol)
	dim(v) <- c(nrow, ncol)
	v
}


#difference wrt kfoots: models can be just one, matrix is always gapmat
lLik2 <- function(v, gapmat, model, ucs, mConst, nthreads = 1){
	#s1 <- getRealSeconds()
	#nbinom_llik(model$mu, model$r, ucs$values, tmp, nthreads)
	#s2 <- getRealSeconds()
	multinom_llik(gapmat, model$ps, v, ucs$map, tmp, mConst, nthreads)
	#s3 <- getRealSeconds()
	
	#cat("threads: " "nbinom: ", s2-s1, " multinom: ", s3-s2, "\n")
	
	
}



#difference wrt kfoots: models can be just one, matrix is always gapmat
lLik <- function(llik, gapmat, model, ucs = NULL, mConst = NULL, scanReverse = FALSE, nthreads = 1){
	if (!scanReverse){
		lLikGapMat(gapmat, list(model), ucs, mConst, llik, nthreads)
	} else {
		imodel <- model; imodel$ps <- rev(model$ps)
		lLikGapMat(gapmat, list(model, imodel), ucs, mConst, llik, nthreads)
	}
}

#fitModelFromColumns(SEXP gapmat, Rcpp::List model, Rcpp::List ucs, int negstrand = 0, int nthreads=1)
#difference wrt kfoots: gapmat and always one model
fitModel <- function(gapmat, ucs = NULL, oldModel = NULL, reverse = 0, forceSym=FALSE, nthreads = 1){
	if (is.null(oldModel)) {
		oldModel <- list(mu=-1, r=-1, ps=rep(-1, gapmat$nrow))
	}

	mod <- fitModelFromColumns(gapmat, oldModel, ucs=ucs, nthreads=nthreads, negstrand=reverse)
	if (forceSym) mod$ps <- (mod$ps + rev(mod$ps))/2

	mod
}

gibbs = function(initmat, footlen, colset, nmotifs, tol = 1e-8, maxiter = 100, show = TRUE, forceSym = FALSE, scanReverse=FALSE, overlap=TRUE, nthreads = 1){
  if (is.vector(initmat)){
    initmat = matrix(initmat, nrow=1)
  }
  
  #do not need to scan the two strands if you only have symmetric models
  if (forceSym) scanReverse <- FALSE
  
  #it wouldn't take long to adapt the code for overlaps of different type, but for now it's not needed
  if (overlap) overlap <- 0
  else overlap <- as.integer(footlen/nrow(initmat))
  
  #current column set
  if (!inherits(colset, "integer")) stop("invalid column set provided")
  if (max(colset) > ncol(initmat) - footlen/nrow(initmat) + 1 || min(colset) <= 0) stop("invalid column indices provided")
  
  
  storage.mode(initmat) <- "integer"
  storage.mode(colset) <- "integer"
  
  #columns should be sorted and duplicate columns don't make sense
  colset <- mapToUnique(colset)$values
  
  
  motifs = list()
  #preprocessing done once here
  swmat = asSWMat(initmat, nrow(initmat), footlen)
  cs = colSumsIntSW(swmat, nthreads)
  initucs = mapToUnique(cs)
  
  
  #manual profiling
  fitTime <- 0
  lLikTime <- 0
  allocTime <- 0
  subsetTime <- 0
  zscoreTime <- 0
  filterTime <- 0
  updateTime <- 0
  totTime <- getRealSeconds()
  
  
  for(motif in 1:nmotifs){
    nloci = length(colset)
    cat(nloci, "\n")
    
    ## subset matrix and preprocessing data
    
    tmp <- getRealSeconds()
    bgmat = asGapMat(initmat, colset, footlen)
    bgucs = subsetM2U(initucs, colset)
    subsetTime <- subsetTime + getRealSeconds() - tmp
    
    
    #allocating memory
    tmp <- getRealSeconds()
    bg <- numeric(length(bgmat$colset))
    if (scanReverse) {
      lnL <- createZeroMatrix(2, length(bgmat$colset))
    } else {
      lnL <- numeric(length(bgmat$colset))
    }
    allocTime <- allocTime + getRealSeconds() - tmp
    
    #background log-likelihood, used a constant to subtract from the model log-likelihood
    #the background model is anyway symmetric
    
    tmp <- getRealSeconds()
    model = fitModel(bgmat, ucs = bgucs, forceSym=T, nthreads = nthreads)
    fitTime <- fitTime + getRealSeconds() - tmp
    
    
    
    tmp <- getRealSeconds()
    lLik(bg, bgmat, model, ucs = bgucs, mConst = bg, nthreads = nthreads)
    lLikTime <- lLikTime + getRealSeconds() - tmp
    
    bg <- -bg
    
    #to parallelize some functions efficiently
    breaks <- findBreaks(colset, overlap, nthreads)
    
    ## sample some
    if (!scanReverse) {
      w = sample(colset, as.integer(nloci / 100))
      reverse = 0
    } else {
      w = sample(colset, as.integer(nloci / 50))
      reverse = as.integer(length(w)/2)
    }
    
    
    old_model = NULL
    converged = FALSE
    for (iter in 1:maxiter){
      ## subset matrix and preprocessing data
      
      tmp <- getRealSeconds()
      mat = asGapMat(initmat, w, footlen)
      ucs = subsetM2U(initucs, w)
      subsetTime <- subsetTime + getRealSeconds() - tmp
      
      tmp <- getRealSeconds()
      new_model = fitModel(mat, ucs=ucs, oldModel = old_model, forceSym=forceSym, nthreads = nthreads, reverse=reverse)
      fitTime <- fitTime + getRealSeconds() - tmp
      #cat("model, mu: ", new_model$mu, " r: ", new_model$r, "ps summary:\n")
      #print(summary(new_model$ps))
      
      converged = !is.null(old_model) && compareModels(old_model, new_model, tol)
      #cat("converged: ", converged, "\n")

      if (is.na(converged)){
        stop("converged gave na")
      }

      #plot(model$ps[1:43], type = 'l', main = iter, ylim = range(model$ps)); lines(model$ps[44:86], col = 2)
      #plotFootLine(new_model$ps*new_model$mu, main = paste("motif ", motif, "iteration: ", iter, sep = ""), strandIsCol=F)
      #plot(new_model$ps, type = 'l', main = paste("motif ", motif, "iteration: ", iter, sep = ""))
      
      tmp <- getRealSeconds()
      lLik(lnL, bgmat, new_model, ucs = bgucs, mConst = bg, scanReverse = scanReverse, nthreads = nthreads)
      lLikTime <- lLikTime + getRealSeconds() - tmp
      
      #cat("w lLik done, summary:\n")
      #print(summary(lnL))
      
      tmp <- getRealSeconds()
      thresh = zScoreThresh(lnL, 2, nthreads=nthreads)
      zscoreTime <- zscoreTime + getRealSeconds() - tmp
      
      if (is.nan(thresh)) break #fails because the model gives nan or -inf fore some loci, it's a weird model with some ps=0

      #when overlap=T and scanReverse=F it just does w = colset[lnL > thresh]
      tmp <- getRealSeconds()
      filtered = filter(colset, lnL, thresh, overlap)
      filterTime <- filterTime + getRealSeconds() - tmp
      
      tmp <- getRealSeconds()
      w = filtered$idxs
      reverse = filtered$reverse
      updateTime <- updateTime + getRealSeconds() - tmp
      
      
      if (length(w) < 2) break #failed because not enough loci are above the threshold
      
      if (converged == TRUE) break

      old_model = new_model
    }
    if (converged == TRUE){
      motifs[[motif]] = new_model
      motifs[[motif]]$w = w
      motifs[[motif]]$reverse = reverse
      if (show) {
        x11()
        plotFootLine(new_model$ps*new_model$mu, main = paste("motif ", motif, sep = ""), strandIsCol=F)
      }
    }
    radius <- max(1, overlap)
    colset = removeOverlapping(colset, w, radius)
  }
  
  
  totTime <- getRealSeconds() - totTime
  
  cat("total time: ", totTime, "\nfit model time: ", fitTime, "\nlog lik time: ", lLikTime,"\nalloc time: ", allocTime,"\nsubset time: ", subsetTime,"\nupdate time: ", updateTime,
      "\nzscore time: ", zscoreTime,"\nfilter time: ", filterTime, "\n")
   
  motifs
}

