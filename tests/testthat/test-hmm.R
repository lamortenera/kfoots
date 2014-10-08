context("HMM and mixture models")

#generate a random vector of length l
#such that the values (all non-negative) sum up to 1
rndPS <- function(l){
	ps <- runif(l)
	ps/sum(ps)
}

test_that("hmm functions work",{
	if (require("HMM")){
		nstat <- 10
		nsymb <- 10
		seqlen <- c(1000, 100, 10)
		#make the HMM object
		States <- paste0("state_", 1:nstat)
		Symbols <- paste0("symbol_", 1:nsymb)
		transProbs <- t(sapply(rep(nstat, nstat), rndPS))
		emissionProbs <- t(sapply(rep(nsymb, nstat), rndPS))
		startProbs <- rndPS(nstat)
		hmm <- initHMM(States, Symbols, startProbs, transProbs, emissionProbs)
		#make the observations
		obs <- sample(nsymb, sum(seqlen), replace=T)
		#get their posteriors
		seqstart <- cumsum(c(1,seqlen))[1:length(seqlen)]
		postList <- lapply(1:length(seqlen), function(idx){
			s <- seqstart[idx]
			e <- s + seqlen[idx] - 1
			HMM:::posterior(hmm, obs[s:e])
		})
		their_post <- do.call(cbind, postList)
		#get their new init probabilities
		their_initP <- their_post[,seqstart]
		
		#get their new transition probabilities
		transList <- lapply(1:length(seqlen), function(idx){
			s <- seqstart[idx]
			e <- s + seqlen[idx] - 1
			HMM:::baumWelchRecursion(hmm, obs[s:e])$TransitionMatrix
		})
		their_exptrans <- prop.table(Reduce("+", transList),1)
		#get their total log-likelihood
		their_llik <- sum(sapply(1:length(seqlen), function(idx){
			s <- seqstart[idx]
			e <- s + seqlen[idx] - 1
			f <- HMM:::forward(hmm, obs[s:e])
			f_lastcol <- f[,ncol(f)]
			llik1 <- max(f_lastcol)
			llik2 <- log(sum(exp(f_lastcol-llik1)))
			llik1+llik2
		}))
		
		#get my posteriors, new initP and new trans
		llik <- log(t(sapply(1:nstat, function(s){ emissionProbs[s, obs] })))
		my_post <- matrix(0, nrow=nstat, ncol=sum(seqlen))
		initP <- matrix(rep(startProbs, length(seqlen)), ncol=length(seqlen))
		fb <- kfoots:::forward_backward(initP, transProbs, llik, seqlen, my_post)
		my_exptrans <- fb$new_trans
		my_llik <- fb$tot_llik
		my_initP <- fb$new_initP
		
		#compare
		dimnames(my_post) <- dimnames(their_post)
		expect_equal(my_post, their_post)
		dimnames(my_exptrans) <- dimnames(their_exptrans)
		expect_equal(my_exptrans, their_exptrans)
		dimnames(my_initP) <- dimnames(their_initP)
		expect_equal(my_initP, their_initP)
		expect_equal(my_llik, their_llik)
		
		#with more threads
		llik <- log(t(sapply(1:nstat, function(s){ emissionProbs[s, obs] })))
		fb <- kfoots:::forward_backward(initP, transProbs, llik, seqlen, my_post, nthreads=20)
		my_exptrans <- fb$new_trans
		my_llik <- fb$tot_llik
		my_initP <- fb$new_initP
		
		#compare
		dimnames(my_post) <- dimnames(their_post)
		expect_equal(my_post, their_post)
		dimnames(my_exptrans) <- dimnames(their_exptrans)
		expect_equal(my_exptrans, their_exptrans)
		dimnames(my_initP) <- dimnames(their_initP)
		expect_equal(my_initP, their_initP)
		expect_equal(my_llik, their_llik)
		
		#test the viterbi algorithm
		vitList <- lapply(1:length(seqlen), function(idx){
			s <- seqstart[idx]
			e <- s + seqlen[idx] - 1
			HMM:::viterbi(hmm, obs[s:e])
		})
		their_viterbi <- unlist(vitList)
		#convert characters to state numbers
		transf <- 1:nstat; names(transf) <- States
		their_viterbi <- transf[their_viterbi]; names(their_viterbi) <- NULL
		llik <- log(t(sapply(1:nstat, function(s){ emissionProbs[s, obs] })))
		my_viterbi <- kfoots:::viterbi(initP, transProbs, llik, seqlen)$vpath
		
		expect_identical(my_viterbi, their_viterbi)
	}
})

test_that("mixture models work", {
	nmod <- 10
	lliks <- matrix(rnorm(1e3*nmod), nrow=nmod)
	mix_coeff <- rndPS(nmod)
	their_post <- exp(log(mix_coeff) + lliks)
	cs <- colSums(their_post)
	their_llik <- sum(log(cs))
	their_post <- their_post/cs[col(their_post)]
	their_new_mix_coeff <- rowMeans(their_post)
	
	my_post <- matrix(0, nrow=nrow(lliks), ncol=ncol(lliks))
	l2p <- llik2posteriors(lliks, mix_coeff, my_post)
	my_llik <- l2p$tot
	my_new_mix_coeff <- l2p$new_mix_coeff
	expect_equal(my_post, their_post)
	expect_equal(my_llik, their_llik)
	expect_equal(my_new_mix_coeff, their_new_mix_coeff)
	#with more threads
	l2p <- llik2posteriors(lliks, mix_coeff, my_post, nthreads=20)
	my_llik <- l2p$tot
	my_new_mix_coeff <- l2p$new_mix_coeff
	expect_equal(my_post, their_post)
	expect_equal(my_llik, their_llik)
	expect_equal(my_new_mix_coeff, their_new_mix_coeff)
	
})
