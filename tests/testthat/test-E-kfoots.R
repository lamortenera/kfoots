context("E: The main functions run as expected")
source("utils.R")

test_that("kfoots works", {
	counts <- exampleData(5000, indep=T)
	
	ks <- list(1,10)
	inits <- list("rnd", "count", "pca")
	nbtypes <- list("dep", "indep", "pois")
	maxiters <- list(10)
	nthreadss <- list(1, 10)
	
	res <- list()
	mcres <- list()
	for (k in ks){
		for (init in inits){
			for (nbtype in nbtypes){
				for (maxiter in maxiters){
					for (nthreads in nthreadss){
						#cat(k, init, nbtype, maxiter, length(seqlens), nthreads, "\n")
						foot <- kfoots(counts, k, maxiter=maxiter, nbtype=nbtype, init=init, nthreads=nthreads, framework="MM", verbose=F)
						models <- foot$models
						expect_true(modelsAreOk(models, k, nrow(counts), nbtype))
						mix_coeff <- foot$mix_coeff
						expect_equal(length(mix_coeff), k)
						expect_equal(sum(mix_coeff), 1)
						expect_true(foot$loglik <= 0)
						expect_true(is.logical(foot$converged))
						llhistory <- foot$llhistory
						expect_true(length(llhistory) <= maxiter)
						expect_true(all(llhistory <= 0))
						posteriors <- foot$posteriors
						expect_equal(dim(posteriors), c(k, ncol(counts)))
						expect_true(all(posteriors >= 0 & posteriors <= 1))
						expect_equal(colSums(posteriors), rep(1, ncol(counts)))
						clusters <- foot$clusters
						expect_equal(length(clusters), ncol(counts))
						expect_true(all(clusters>=1&clusters<=k))
						
						if (init != "rnd"){
							if (nthreads == nthreadss[[1]]) res[[length(res)+1]] <- foot
							else if (nthreads == nthreadss[[2]]) mcres[[length(mcres)+1]] <- foot
						}
					}
				}
			}
		}
	}
	expect_equal(res, mcres, 1e-6)
})



test_that("hmmfoots works", {
	counts <- exampleHMMData(c(1000, 2500, 1500))
	
	ks <- list(1,10)
	seqlenss <- list(5000, c(1, 4999), c(1000, 2500, 1500))
	inits <- list("rnd", "count", "pca")
	nbtypes <- list("dep", "indep", "pois")
	maxiters <- list(10)
	nthreadss <- list(1, 10)
	
	res <- list()
	mcres <- list()
	for (k in ks){
		for (seqlens in seqlenss){
			for (init in inits){
				for (nbtype in nbtypes){
					for (maxiter in maxiters){
						for (nthreads in nthreadss){
							#cat(k, init, nbtype, maxiter, length(seqlens), nthreads, "\n")
							hmm <- kfoots(counts, k, maxiter=maxiter, nbtype=nbtype, init=init, seqlens=seqlens, nthreads=nthreads, framework="HMM", verbose=F)
							models <- hmm$models
							expect_true(modelsAreOk(models, k, nrow(counts), nbtype))
							trans <- hmm$trans
							expect_equal(dim(trans), c(k,k))
							expect_equal(rowSums(trans), rep(1, k))
							initP <- hmm$initP
							expect_equal(dim(initP), c(k, length(seqlens)))
							expect_equal(colSums(initP), rep(1, length(seqlens)))
							expect_true(hmm$loglik <= 0)
							expect_true(is.logical(hmm$converged))
							llhistory <- hmm$llhistory
							expect_true(length(llhistory) <= maxiter)
							expect_true(all(llhistory <= 0))
							posteriors <- hmm$posteriors
							expect_equal(dim(posteriors), c(k, ncol(counts)))
							expect_true(all(posteriors >= 0 & posteriors <= 1))
							expect_equal(colSums(posteriors), rep(1, ncol(counts)))
							clusters <- hmm$clusters
							expect_equal(length(clusters), ncol(counts))
							expect_true(all(clusters>=1&clusters<=k))
							viterbi <- hmm$viterbi
							expect_equal(length(viterbi$vpath), ncol(counts))
							expect_true(all(viterbi$vpath >=1 & viterbi$vpath <=k))
							eq <- all.equal(viterbi$vllik, hmm$loglik)
							if (eq != T) expect_true(viterbi$vllik < hmm$loglik)
							
							if (init != "rnd"){
								label <- paste0("k_", k, ",init_", init, ",nbtype_", nbtype, ",slens_", paste(seqlens, collapse=";"))
								#we don't test that the viterbi paths coincide, because
								#there are many equivalent viterbi paths and the choice
								#depends a lot on the numerical fuzz
								hmm$viterbi <- NULL
								if (nthreads == nthreadss[[1]]) res[[label]] <- hmm
								else if (nthreads == nthreadss[[2]]) mcres[[label]] <- hmm
							}
						}
					}
				}
			}
		}
	}
	expect_equal(res, mcres, 1e-6)
	#bug <- list(res=res, mcres=mcres, counts=counts)
	#save(bug, file="/project/ale/home/data/kfoots_pkg/bug.Rdata")
})

