#include <Rcpp.h>

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
List forward_backward(NumericVector initP, NumericMatrix trans, NumericMatrix lliks, NumericVector seqlens){
	int k = initP.length();
	int ncol = lliks.ncol();
	NumericMatrix posteriors(k, ncol);
	NumericMatrix new_trans(k, k);
	NumericMatrix tmp(k, k);
	NumericVector backward(k);
	NumericVector new_backward(k);
	
	/* transform the lliks matrix to the original space (exponentiate). 
	 * A column-specific factor can be multiplied to obtain a better numerical
	 * stability */
	
	double tot_llik = 0;
	for (int c = 0; c < ncol; ++c){
		MatrixColumn<REALSXP> llikcol = lliks.column(c);
		/* get maximum llik in the column */
		double max_llik = llikcol[0];
		for (int r = 1; r < k; ++r){
			if (llikcol[r] > max_llik){ max_llik = llikcol[r]; }
		}
		/* subtract maximum and exponentiate */
		tot_llik += max_llik;
		for (int r = 0; r < k; ++r){
			llikcol[r] = exp(llikcol[r] - max_llik);
		}
	}
	
	/* Do forward and backward loop for each chunk (defined by seqlens) */
	for (int o = 0, chunk_start = 0; o < seqlens.length(); chunk_start += seqlens[o], ++o){
		int chunk_end = chunk_start + seqlens[o];
		/* FORWARD LOOP */
		/* first iteration is from fictitious start state */
		double cf = 0;
		MatrixColumn<REALSXP> emissprob0 = lliks.column(chunk_start);
		MatrixColumn<REALSXP> forward0 = posteriors.column(chunk_start);
		for (int r = 0; r < k; ++r){
			double p = emissprob0[r]*initP[r];
			forward0[r] = p;
			cf += p;
		}
		for (int r = 0; r < k; ++r){
			forward0[r] = forward0[r]/cf;
		}
		tot_llik += log(cf);
		/* all other iterations */
		for (int i = chunk_start + 1; i < chunk_end; ++i){
			cf = 0;//scaling factor
			MatrixColumn<REALSXP> emissprob = lliks.column(i);
			MatrixColumn<REALSXP> forward = posteriors.column(i);
			MatrixColumn<REALSXP> last_forward = posteriors.column(i-1);
		
			for (int t = 0; t < k; ++t){
				for (int s = 0; s < k; ++s){
					forward[t] += last_forward[s]*trans(s,t)*emissprob[t];
				}
				cf += forward[t];
			}
			for (int t = 0; t < k; ++t){
				forward[t] = forward[t]/cf;
			}
			tot_llik += log(cf);
		}
		
		/* BACKWARD LOOP */
		/* first iteration set backward to 1/k, 
		 * last column of posteriors is already ok */
		for (int r = 0; r < k; ++r){
			backward[r] = 1.0/k;
		}
		for (int i = chunk_end-2; i >= chunk_start; --i){
			MatrixColumn<REALSXP> emissprob = lliks.column(i+1);
			MatrixColumn<REALSXP> posterior = posteriors.column(i);
			cf = 0;
			double norm = 0;
			/* joint probabilities and backward vector */
			for (int s = 0; s < k; ++s){
				//the forward variable is going to be overwritten with the posteriors
				double pc = posterior[s];
				new_backward[s] = 0;
				for (int t = 0; t < k; ++t){
					double p = trans(s,t)*emissprob[t]*backward[t];
					tmp(s, t) = pc*p;
					new_backward[s] += p;
				}
				cf += new_backward[s];
			}
			/* update backward vector */
			for (int s = 0; s < k; ++s){
				backward[s] = new_backward[s]/cf;
				norm += backward[s]*posterior[s];
			}
			/* update transition probabilities */
			for (int t = 0; t < k*k; ++t){
				new_trans[t] += tmp[t]/(norm*cf);
			}
			/* get posteriors */
			for (int s = 0; s < k; ++s){
				posterior[s] = posterior[s]*backward[s]/norm;
			}
		}
	}

	/* normalizing new_trans matrix */
	for (int c = 0; c < k; ++c){
		MatrixRow<REALSXP> row = new_trans.row(c);
		row = row/sum(row);
	}
	
	return List::create(_("posteriors")=posteriors, _("tot_llik")=tot_llik, _("new_trans")=new_trans);
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
		vpath[chunk_end - 1] = maxp;
		for (int i = chunk_end - 2; i >= chunk_start; --i){
			maxp = backtrack(maxp, i - chunk_start + 1);
			vpath[i] = maxp + 1; //in R indices are 1-based
		}
	}
	return List::create(_("vpath")=vpath, _("vllik")=tot_maxscore);
}
