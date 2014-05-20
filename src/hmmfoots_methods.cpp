#include <Rcpp.h>
#include "core.hpp"

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
			
			scores = new_scores;
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
