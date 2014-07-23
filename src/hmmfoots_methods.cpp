#include <Rcpp.h>
#include "core.cpp"


static inline double forward_backward_core(Vec<double> initP, Mat<double> trans, Mat<double> lliks, Vec<int> seqlens, Mat<double> posteriors, Mat<double> new_trans, int nthreads){
	nthreads = std::max(1, nthreads);
	
	int nrow = lliks.nrow;
	int ncol = lliks.ncol;
	int nchunk = seqlens.len;
	
	//temporary objects 
	std::vector<int> chunk_startsSTD(seqlens.len, 0);
	Vec<int> chunk_starts = asVec<int>(chunk_startsSTD);
	//get the start of each chunk
	for (int i = 0, acc = 0; i < nchunk; ++i){chunk_starts[i] = acc; acc += seqlens[i];}
	
	long double tot_llik = 0;
	
	
	#pragma omp parallel num_threads(nthreads)
	{
		//each thread gets one copy of these temporaries
		std::vector<double> tmpSTD(nrow*nrow, 0); 
		std::vector<double> thread_new_transSTD(nrow*nrow, 0); 
		std::vector<double> backward(nrow, 0);
		std::vector<double> new_backward(nrow, 0);
		long double thread_llik = 0;
		
		Mat<double> tmp = asMat(tmpSTD, nrow);
		Mat<double> thread_new_trans = asMat(thread_new_transSTD, nrow);
		
		/* transform the lliks matrix to the original space (exponentiate). 
		 * A column-specific factor is multiplied to obtain a better numerical
		 * stability */
		#pragma omp for schedule(static) reduction(+:tot_llik)
		for (int c = 0; c < ncol; ++c){
			double* llikcol = lliks.colptr(c);
			/* get maximum llik in the column */
			double max_llik = llikcol[0];
			for (int r = 1; r < nrow; ++r){
				if (llikcol[r] > max_llik){ max_llik = llikcol[r]; }
			}
			/* subtract maximum and exponentiate */
			tot_llik += max_llik;
			for (int r = 0; r < nrow; ++r, ++llikcol){
				*llikcol = exp(*llikcol - max_llik);
			}
		}
		
		/* Do forward and backward loop for each chunk (defined by seqlens)
		 * Chunks might have very different lengths (that's why dynamic schedule). */
		#pragma omp for schedule(dynamic,1) nowait
		for (int o = 0; o < nchunk; ++o){
			int chunk_start = chunk_starts[o];
			int chunk_end =  chunk_start + seqlens[o];
			
			/* FORWARD LOOP */
			/* first iteration is from fictitious start state */
			{
				double cf = 0;//scaling factor
				double* emissprob = lliks.colptr(chunk_start);
				double* forward = posteriors.colptr(chunk_start);
				for (int r = 0; r < nrow; ++r){
					double p = emissprob[r]*initP[r];
					forward[r] = p;
					cf += p;
				}
				for (int r = 0; r < nrow; ++r){
					forward[r] = forward[r]/cf;
				}
				thread_llik += log(cf);
			}
			/* all other iterations */
			for (int i = chunk_start + 1; i < chunk_end; ++i){
				double cf = 0;//scaling factor
				double* emissprob = lliks.colptr(i);
				double* forward = posteriors.colptr(i);
				double* last_forward = posteriors.colptr(i-1);
			
				for (int t = 0; t < nrow; ++t){
					double* transcol = trans.colptr(t);
					double acc = 0;
					for (int s = 0; s < nrow; ++s){
						acc += last_forward[s]*transcol[s];
					}
					acc *= emissprob[t];
					forward[t] = acc;
					cf += acc;
				}
				for (int t = 0; t < nrow; ++t){
					forward[t] = forward[t]/cf;
				}
				thread_llik += log(cf);
			}
			
			/* BACKWARD LOOP */
			/* first iteration set backward to 1/k, 
			 * last column of posteriors is already ok */
			for (int r = 0; r < nrow; ++r){
				backward[r] = 1.0/nrow;
			}
			for (int i = chunk_end-2; i >= chunk_start; --i){
				double* emissprob = lliks.colptr(i+1);
				double* posterior = posteriors.colptr(i);
				double cf = 0;
				double norm = 0;
				/* joint probabilities and backward vector */
				for (int s = 0; s < nrow; ++s){
					//the forward variable is going to be overwritten with the posteriors
					double pc = posterior[s];
					double acc = 0;
					
					for (int t = 0; t < nrow; ++t){
						double p = trans(s, t)*emissprob[t]*backward[t];
						tmp(s, t) = pc*p;
						acc += p;
					}
					
					new_backward[s] = acc;
					cf += acc;
				}
				/* update backward vector */
				for (int s = 0; s < nrow; ++s){
					backward[s] = new_backward[s]/cf;
					norm += backward[s]*posterior[s];
				}
				/* update transition probabilities */
				for (int t = 0, e = nrow*nrow; t < e; ++t){
					thread_new_trans[t] += tmp[t]/(norm*cf);
				}
				/* get posteriors */
				for (int s = 0; s < nrow; ++s){
					posterior[s] = posterior[s]*backward[s]/norm;
				}
			}
		}
		//protected access to the shared variables
		#pragma omp critical
		{
			tot_llik += thread_llik;
			for (int p = 0, q = nrow*nrow; p < q; ++p){
				new_trans[p] += thread_new_trans[p];
			}
		}
	}

	/* normalizing new_trans matrix */
	// should I put it inside the parallel region?
	// The parallelization overhead might take longer than
	// this loop....
	for (int row = 0; row < nrow; ++row){
		double sum = 0;
		for (int col = 0; col < nrow; ++col){sum += new_trans(row, col);}
		for (int col = 0; col < nrow; ++col){new_trans(row, col)/=sum;}
	}
	
	return (double) tot_llik;
}


using namespace Rcpp;

typedef NumericVector::iterator diter;
typedef IntegerVector::iterator iiter;

//' Forward-backward algorithm
//'
//' Forward-backward algorithm using the scaling technique.
//' That's more stable (and maybe even faster) than the method with the logarithm.
//' Warning: this function overwrites the lliks matrix. This is probably a bad idea
//' because normally in the last loop you want to use the same matrix
//' for forward_backward and viterbi. I might change that in the future.
//' @param initP vector of initial probabilities
//' @param trans transition matrix (rows are previous state, columns are next state)
//' @param lliks matrix with emission probabilities for each datapoint and each state.
//' Columns are datapoints and rows are states.
//' @param seqlens length of each subsequence of datapoints (set this to ncol(lliks)
//' if there is only one sequence).
//' @return a list with the following arguments:
//'	\item{posteriors}{posterior probability of being in a certain state for a certain datapoint}
//'	\item{tot_llik}{total log-likelihood of the data given the hmm model}
//'	\item{new_trans}{update for the transition probabilities (it is already normalized)}
//' @export
// [[Rcpp::export]]
List forward_backward(NumericVector initP, NumericMatrix trans, NumericMatrix lliks, IntegerVector seqlens, NumericMatrix posteriors, int nthreads=1){
	int nmod = initP.length();
	double totlen = Rcpp::sum(seqlens);
	if (nmod != trans.nrow() || nmod != trans.ncol() || nmod != lliks.nrow() || nmod != posteriors.nrow()) Rcpp::stop("Unable to figure out the number of models");
	if (((double) lliks.ncol()) != totlen || ((double)posteriors.ncol()) != totlen) Rcpp::stop("Seqence lengths don't match with the provided matrices");
	
	NumericMatrix newTrans(trans.nrow(), trans.ncol());
	double tot_llik = forward_backward_core(asVec(initP), asMat(trans), asMat(lliks), asVec(seqlens), asMat(posteriors), asMat(newTrans), nthreads);
	return List::create(_("posteriors")=posteriors, _("tot_llik")=tot_llik, _("new_trans")=newTrans);
}

//' Viterbi algorithm
//'
//' Standard viterbi algorithm in the log space
//' @param initP vector of initial probabilities
//' @param trans transition matrix (rows are previous state, columns are next state)
//' @param lliks matrix with emission probabilities for each datapoint and each state.
//' Columns are datapoints and rows are states.
//' @param seqlens length of each subsequence of datapoints (set this to ncol(lliks)
//' if there is only one sequence).
//' @return a list with the following arguments:
//'	\item{vpath}{viterbi path}
//'	\item{vllik}{log-likelihood of the viterbi path}
//' @export
// [[Rcpp::export]]
List viterbi(NumericVector initP, NumericMatrix trans, NumericMatrix lliks, NumericVector seqlens){
	int nmod = initP.length();
	double totlen = Rcpp::sum(seqlens);
	if (nmod != trans.nrow() || nmod != trans.ncol() || nmod != lliks.nrow()) Rcpp::stop("Unable to figure out the number of models");
	if (((double) lliks.ncol()) != totlen) Rcpp::stop("Seqence lengths don't match with the provided matrix");
	
	int k = initP.length();
	int ncol = lliks.ncol();
	NumericVector vpath(ncol);
	IntegerMatrix backtrack(k, max(seqlens));
	NumericVector scores(k);
	NumericVector new_scores(k);
	
	/* log-transform the transition probabilities */
	NumericMatrix ltrans(k,k);
	for (diter curr = ltrans.begin(), currt = trans.begin(); curr < ltrans.end(); ++curr, ++currt){
		*curr = log(*currt);
	}
	
	/* Viterbi independently on each chunk */
	double tot_maxscore = 0;
	for (int o = 0, chunk_start = 0; o < seqlens.length(); chunk_start += seqlens[o], ++o){
		int chunk_end = chunk_start + seqlens[o];
		/* dynamic programming */
		scores = lliks.column(chunk_start) + log(initP);
		for (int i = chunk_start + 1; i < chunk_end; ++i){
			
			MatrixColumn<REALSXP> llikcol = lliks.column(i);
			MatrixColumn<INTSXP> backtrackcol = backtrack.column(i-chunk_start);
			
			for (int t = 0; t < k; ++t){
				int maxs = 0;
				double maxscore = scores[0] + ltrans(0, t);
				for (int s = 1; s < k; ++s){
					double currscore = scores[s] + ltrans(s,t);
					if (currscore > maxscore){
						maxscore = currscore;
						maxs = s;
					}
				}
				backtrackcol[t] = maxs;
				new_scores[t] = llikcol[t] + maxscore;
			}
			
			memcpy(scores.begin(), new_scores.begin(), sizeof(double)*k);
		}
		
		/* backtracking */
		int maxp = 0;
		double maxscore = scores[0];
		for (int p = 1; p < k; ++p){
			if (scores[p] > maxscore){
				maxscore = scores[p];
				maxp = p;
			}
		}
		tot_maxscore += maxscore;
		vpath[chunk_end - 1] = maxp + 1;
		for (int i = chunk_end - 2; i >= chunk_start; --i){
			maxp = backtrack(maxp, i - chunk_start + 1);
			vpath[i] = maxp + 1; //in R indices are 1-based
		}
	}
	return List::create(_("vpath")=vpath, _("vllik")=tot_maxscore);
}


// [[Rcpp::export]]
Rcpp::IntegerVector orderColumns(Rcpp::IntegerMatrix mat){
	Rcpp::IntegerVector order(mat.ncol());
	orderColumns_core(asMat(mat), asVec(order));
	return order;
}
