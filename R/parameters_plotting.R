plotMatrix <- function(mat, xticks=colnames(mat), yticks=rownames(mat), zlim=range(mat, na.rm=TRUE), col=heat.colors(12), sides=c(1,2),...){
	image(t(mat[nrow(mat):1,]), zlim=zlim, col=col, xaxt="n", yaxt="n",...)
	
	if (is.null(xticks)){
		#user selected NULL or there are no colnames in the matrix
		xticks = 1:ncol(mat)
	}
	if (!is.na(xticks)[1]){
		if (ncol(mat)==1)
			at = 0
		else
			at = (0:(ncol(mat)-1))/(ncol(mat)-1)
		axis(side=sides[1], at=at, labels=xticks, tick=F)
	}
	
	if (is.null(yticks)){
		#user selected NULL or there are no rownames in qthe matrix
		yticks = 1:nrow(mat)
	}
	if (!is.na(yticks)[1]){
		if (nrow(mat)==1)
			at = 0
		else
			at = ((nrow(mat)-1):0)/(nrow(mat)-1)
		axis(side=sides[2], at=at, labels=yticks, tick=F)
	}
}

plotColorKey <- function(zlim, n=20, col=heat.colors(n), side=4, lab="color key", ...){
	z <- seq(zlim[1], zlim[2], length.out=n)
	if (side %% 2 == 0){
		x <- 0
		y <- z
		mat <- t(as.matrix(z))
	} else {
		y <- 0
		x <- z
		mat <- as.matrix(z)
	}
	image(x,y,mat, xaxt="n", yaxt="n",col=col,xlab=NA, ylab=NA, ...)
	axis(side=side)
	if (!is.null(lab) && !is.na(lab)[1]){
		mtext(lab, side=side, line=2)
	}
	
}


getMeanMatrix <- function(models){
	ps <- getPS(models)
	mus <- getNBs(models)[1,]
	
	t(ps) * mus
 }


reorderMat <- function(models){
	mps <- log(getMeanMatrix(models))
	if (min(mps)==-Inf){
	#replacing -Inf with a reasonable low value
		tmpmps <- mps
		mx <- max(mps) #max value
		tmpmps[tmpmps==-Inf] <- max(mps)
		mn <- min(tmpmps) #lowest value when there are no -Inf
		mps[mps==-Inf] <- mn - (mx - mn)/length(mps) #pseudocount: a bit less than the minumum
	}
	list(colIdx=hclust(dist(mps))$order, rowIdx=hclust(dist(t(mps)))$order)
}

getPS <- function(models){
	if (is.null(names(models)))
		names(models) <- paste0("cluster ", 1:length(models))
	do.call(cbind, lapply(models, function(m) m$ps))
}

getNBs <- function(models){
	if (is.null(names(models)))
		names(models) <- paste0("cluster ", 1:length(models))
	do.call(cbind, lapply(models, function(m) c(m$mu, m$r)))
}

plotNBs <- function(nbs, eps=0.2, xlab="count",...){
	nbs <- nbs[,ncol(nbs):1]
	ys <- seq(0,1, length.out=ncol(nbs))
	xs <- nbs[1,]
	sds <- sqrt(xs + (xs^2)/nbs[2,])
	xlim <- c(min(xs-sds), max(xs+sds))
	spacer <- 1/(2*(ncol(nbs)-1))
	ylim <- c(0-spacer,1+spacer)
	par(yaxs="i")
	plot(NULL, xlim=xlim, ylim=ylim, yaxt="n", ylab=NA, xlab=xlab, ...)
	axis(side=2, tick=F, at=ys, labels=colnames(nbs))
	X <- as.numeric(rbind(xs-sds, xs+sds, NA))
	Y <- as.numeric(rbind(ys, ys, NA))
	lines(X, Y)
	X <- as.numeric(rbind(xs-sds, xs-sds, NA))
	Y <- as.numeric(rbind(ys - spacer*eps, ys + spacer*eps, NA))
	lines(X, Y)
	X <- as.numeric(rbind(xs+sds, xs+sds, NA))
	lines(X, Y)
	points(xs, ys)
}

plotModels <- function(models, mix_coeff, widths=c(0.2,0.5,0.2,0.1)){
	os <- reorderMat(models)
	models <- models[os$colIdx]
	mix_coeff <- mix_coeff[os$colIdx]
	
	if (is.null(names(models)))
		names(models) <- paste0("cluster ", 1:length(models))
	
	
	
	ps <- getPS(models)[os$rowIdx,]
	nbs <- getNBs(models)
	layout(matrix(c(1,2,3,4), nrow=1), widths=widths)
	par(mar=c(5,4,4,1))
	plotNBs(nbs, main="mean and sd", xlab="tot read count")
	library(fields)
	col <- tim.colors(100)
	par(mar=c(5,1,4,1), las=2)
	plotMatrix(t(ps), main="multinomial ps", yticks=NA, col=col)
	par(mar=c(5,1,4,1))
	plotMatrix(as.matrix(mix_coeff), main="mix. coeff.", zlim=c(0,1), yticks=NA, xticks=NA, col=col)
	par(mar=c(5,1,4,3.5), las=0)
	plotColorKey(c(0,1), col=col, n=64)
	
}


plotModels2 <- function(models, mix_coeff, widths=c(0.3,0.5,0.2), heights=c(0.1,0.9)){
	os <- reorderMat(models)
	models <- models[os$colIdx]
	mix_coeff <- mix_coeff[os$colIdx]
	
	if (is.null(names(models)))
		names(models) <- paste0("cluster ", 1:length(models))
	
	
	library(fields)
	col <- tim.colors(100)
	mns <- getMeanMatrix(models)[,os$rowIdx]
	nbs <- getNBs(models)
	layout(matrix(c(1,2,3,4,5,6), nrow=2), widths=widths, heights=heights)
	
	
	umar <- 2
	colkeylowmar <- 2
	lmar <- 1
	rmar <- 1
	hmodlabelmar <- 6
	clustlabelmar <- 6
	middlemar <- 1
	
	#NB params title
	par(mar=c(0, clustlabelmar, umar+1, 1))
	plot.new()
	title(main="Tot read count\nmean and sd")
	#mtext(, side=3, line=0)
	
	#NB params
	par(mar=c(hmodlabelmar, clustlabelmar,1,1), las=1)
	plotNBs(nbs, xlab="tot read count")
	
	#Means matrix color key
	mnsrange <- range(mns)
	par(mar=c(colkeylowmar,lmar,umar,rmar))
	plotColorKey(mnsrange, col=col, n=64, side=1, main="Mean counts", lab=NULL)
	
	#Means matrix
	par(mar=c(hmodlabelmar,lmar,middlemar,rmar), las=2)
	plotMatrix(mns, yticks=NA, col=col)
	
	#mix coeff color key
	mcrange <- range(mix_coeff)
	par(mar=c(colkeylowmar,lmar,umar,rmar), las=1)
	plotColorKey(mcrange, col=col, n=64, side=1, main="Mix. coeff.", lab=NULL)
	
	#mix coeff
	par(mar=c(hmodlabelmar,lmar,middlemar,rmar))
	plotMatrix(as.matrix(mix_coeff), yticks=NA, xticks=NA, col=col)
}


plotHMM <- function(models, trans, widths=c(0.2,0.35,0.35,0.1)){
	os <- reorderMat(models)
	models <- models[os$colIdx]
	trans <- trans[os$colIdx, os$colIdx]
	
	
	if (is.null(names(models)))
		names(models) <- paste0("cluster ", 1:length(models))
	
	
	rownames(trans) <- names(models)
	colnames(trans) <- names(models)
	ps <- getPS(models)[os$rowIdx,]
	nbs <- getNBs(models)
	lmar = 5
	umar = 4
	library(fields)
	col <- tim.colors(100)
	layout(matrix(c(1,2,3,4), nrow=1), widths=widths)
	par(mar=c(lmar,4,umar,1))
	plotNBs(nbs, main="mean and sd", xlab="tot read count")
	par(mar=c(lmar,1,umar,1), las=2)
	plotMatrix(t(ps), main="multinomial probs.", zlim=c(0,1), yticks=NA, col=col)
	par(mar=c(lmar,1,umar,1))
	plotMatrix(trans, main="trans. probs.", zlim=c(0,1), yticks=NA, col=col)
	par(mar=c(lmar,1,umar,3.5), las=0)
	plotColorKey(c(0,1), col=col, n=64)
	
}

plotHMM2 <- function(models, trans, widths=c(0.3,0.35,0.35), heights=c(0.1,0.9), col=NULL){
	os <- reorderMat(models)
	models <- models[os$colIdx]
	trans <- trans[os$colIdx, os$colIdx]
	
	if (is.null(names(models)))
		names(models) <- paste0("cluster ", os$colIdx)
	
	if (is.null(col)) {
		library(RColorBrewer)
		col <- brewer.pal(9, "Reds")
	}
	
	mns <- getMeanMatrix(models)[,os$rowIdx]
	nbs <- getNBs(models)
	layout(matrix(c(1,2,3,4,5,6), nrow=2), widths=widths, heights=heights)
	
	
	umar <- 2
	colkeylowmar <- 2
	lmar <- 1
	rmar <- 1
	hmodlabelmar <- 6
	clustlabelmar <- 6
	middlemar <- 1
	
	#NB params title
	par(mar=c(0, clustlabelmar, umar+1, 1))
	plot.new()
	title(main="Tot read count\nmean and sd")
	#mtext(, side=3, line=0)
	
	#NB params
	par(mar=c(hmodlabelmar, clustlabelmar,1,1), las=1)
	plotNBs(nbs, xlab="tot read count")
	
	#Means matrix color key
	mnsrange <- range(mns)
	par(mar=c(colkeylowmar,lmar,umar,rmar))
	plotColorKey(mnsrange, col=col, n=64, side=1, main="Mean counts", lab=NULL)
	
	#Means matrix
	par(mar=c(hmodlabelmar,lmar,middlemar,rmar), las=2)
	plotMatrix(mns, yticks=NA, col=col)
	
	#mix coeff color key
	transrange <- range(trans)
	par(mar=c(colkeylowmar,lmar,umar,rmar), las=1)
	plotColorKey(transrange, col=col, n=64, side=1, main="Transitions", lab=NULL)
	
	#trans
	colnames(trans) <- names(models)
	par(mar=c(hmodlabelmar,lmar,middlemar,rmar), las=2)
	plotMatrix(trans, yticks=NA, col=col)
}

plotModels_old <- function(models, xnames=NA, dendro="both", norm=T,
	Rowv=TRUE, Colv=TRUE, trace="none", col=tim.colors(), mar=c(5,9),...){
	source("/home/wwwcmb/epigen/htdocs/scripts/kmtools.R")
	mat <- getMeanMatrix(models)
	if (is.null(names(models))){
		colnames(mat) <- paste0("cluster ", 1:ncol(mat))
	} else {
		colnames(mat) <- names(models)
	}
	if (length(xnames)==1 && is.na(xnames)){ 
		rownames(mat) <- paste0("C", 1:nrow(mat))
	} else {
		rownames(mat) <- xnames
	}
	if (norm){
		relint = colSums(mat)
		relint = relint/max(relint)
		mat <- apply(mat, 2, function(col) col/sum(col))
		mat = rbind(relint, mat)
		rownames(mat)[1] <- "Tot"
	}
	
	library(gplots)
	library(fields)
	heatmap.2(t(mat), dendro=dendro, Rowv=Rowv, Colv=Colv, trace=trace, col=col, mar=mar,...)
}

plotModelsAsFootprints <- function(models, col=rainbow(length(models)), 
	lines=F,legend=T,lloc="top", ylab="average count",
	mix_coeff=NA, xnames=NA, mar=NA, pch=1,
	...){
	
	mat <- getMeanMatrix(models)
	ylim = c(min(mat), max(mat))
	xs <- 1:nrow(mat)
	
	if (length(xnames)==1 && is.na(xnames)){ 
		plot(NULL, xaxt="none", ylab=ylab, xlab="vector components", xlim=c(1, nrow(mat)), ylim=ylim, col=col[i],...)
		axis(1, at=xs, labels=xs)
	} else {
		if (length(mar)==1 && is.na(mar)){
			lowmar = max(sapply(xnames, nchar))
			mar = c(1+lowmar,4,4,2)
		}
		plot(NULL, xaxt="none", xlab=NA, ylab=ylab, xlim=c(1, nrow(mat)), ylim=ylim, col=col[i], mar=mar, ...)
		axis(1, at=xs, labels=xnames, las=3)
	}
	
	
		
	for (i in 1:length(models)){
		points(xs, mat[,i], col=col[i], pch=pch)
		if (lines){
			lines(xs, mat[,i], col=col[i])
		}
	}
	
	if (legend){
		clab = NA
		if (is.null(names(models))){
			clab = paste("cluster", 1:ncol(mat))
		} else {
			clab = names(models)
		}
		
		if (length(mix_coeff)>1){
			clab = paste0(clab, ", mix. coeff: ", format(mix_coeff, digits=4))
		}
		
		legend(lloc, legend=clab, col=col, pch=pch)
	}
}


plotVector <- function(model, x, y){
	p <- model$ps
	m <- model$mu
	abline(0, p[y]/p[x], lty=2)
	points(m*p[x], m*p[y], pch=19)
}

compareMarginals <- function(counts, models, mix_coeff, quant=.95){
	par(mfcol=c(1, ncol(counts)))
	for (i in 1:ncol(counts)){
		ci = counts[,i]
		maxci = quantile(ci, quant)
		xs <- 0:maxci
		ys <- rep(0, length(xs))
		for (m in 1:length(models)){
			mu = models[[m]]$mu*models[[m]]$ps[i]
			r = models[[m]]$r
			ys <- ys + dnbinom(xs, mu=mu, size=r)*mix_coeff[m]
		}
		cname = names(models[[1]]$ps)[1]
		plot(xs, ys*length(ci), type="l", lty=2, xlab=paste(cname, "counts"), ylab="occurrences")
		points(xs, table(factor(ci, levels=xs)), type="l")
		legend("topright", legend=c("expected", "observed"), lty=c(2, 1))
	}
}

matchClust <- function(clust1, clust2){
	library(lpSolve)
	
	maxlev <- max(clust1, clust2)
	
	f1 <- factor(clust1, levels=1:maxlev)
	f2 <- factor(clust2, levels=1:maxlev)
	
	mat <- table(f1, f2)
	perm <- lp.assign(mat, direction="max")$solution
	minDist <- 1 - sum(mat[as.logical(perm)])/length(clust1)
	
	
	list(distance=minDist, permutation=apply(perm,2,which.max))
}

plotFootLine <- function(l, xlab="base pairs from region start", ylab="average value", legend=T, loc="bottom", strandIsCol = TRUE, ...){
	len <- length(l)/2
	plot(NULL, xlim=c(1, len), ylim=c(0, max(l)), xlab=xlab, ylab=ylab, ...)
	abline(v=(1:len), lty=2, col="gray")
	if (strandIsCol){
		mat <- matrix(l, ncol=2)
		posStrand = mat[,1]
		negStrand = mat[,2]
	} else {
		mat <- matrix(l, nrow=2)
		posStrand = mat[1,]
		negStrand = mat[2,]
	}
	lines(posStrand, col="blue")
	points(posStrand, col="blue")
	lines(negStrand, col="red")
	points(negStrand, col="red")
	
	if (legend){
		legend(loc, legend=c("sense strand", "antisense strand"), col=c("blue", "red"), lty=1, bty="n")
	}
}

