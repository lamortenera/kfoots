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
	if (!is.na(lab)){
		mtext(lab, side=side, line=2)
	}
	
}


getMeanMatrix <- function(models, norm=F){
	mat <- as.matrix(models[[1]]$ps*models[[1]]$mu, ncol=1)
	for (i in 2:length(models)){
		mat <- cbind(mat, models[[i]]$mu*models[[i]]$ps)
	}
	if (norm){
		mat <- apply(mat, 2, function(col) col/sum(col))
	}
	mat
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

plotModels <- function(models, mix_coeff, widths=c(0.2,0.5,0.2,0.1)){
	if (is.null(names(models)))
		names(models) <- paste0("cluster ", 1:length(models))
	ps <- getPS(models)
	nbs <- getNBs(models)
	layout(matrix(c(1,2,3,4), nrow=1), widths=widths)
	par(mar=c(5,4,4,1))
	plotNBs(nbs, main="mean and sd", xlab="tot read count")
	library(fields)
	col <- tim.colors()
	par(mar=c(5,1,4,1))
	plotMatrix(t(ps), main="multinomial ps", zlim=c(0,1), yticks=NA, col=col)
	par(mar=c(5,1,4,1))
	plotMatrix(as.matrix(mix_coeff), main="mix. coeff.", zlim=c(0,1), yticks=NA, xticks=NA, col=col)
	par(mar=c(5,1,4,3.5))
	plotColorKey(c(0,1), col=col, n=64)
	
}

plotHMM <- function(models, trans, widths=c(0.2,0.35,0.35,0.1)){
	if (is.null(names(models)))
		names(models) <- paste0("cluster ", 1:length(models))
	rownames(trans) <- names(models)
	colnames(trans) <- names(models)
	ps <- getPS(models)
	nbs <- getNBs(models)
	lmar = 5
	umar = 4
	library(fields)
	col <- tim.colors()
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
