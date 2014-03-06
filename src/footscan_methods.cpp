#include <Rcpp.h>
#include "core.hpp"
#include <algorithm> 



#define round( d ) ((int) (d + 0.5))


// [[Rcpp::export]]
Rcpp::List asGapMat(Rcpp::IntegerMatrix counts, Rcpp::IntegerVector colset, int nrow){
	Rcpp::IntegerVector idxs(colset.length());
	int* idx = idxs.begin(); int* cls = colset.begin();
	int oldnrow = counts.nrow();
	for (int i = 0, e = colset.length(); i < e; ++i){
		*idx = oldnrow*(*cls  - 1);
		++idx; ++cls;
	}
	
	Rcpp::List ret = Rcpp::List::create(Rcpp::Named("vec")=counts, Rcpp::Named("colset")=idxs, Rcpp::Named("nrow")=nrow);
	ret.attr("class") = "gapmat";
	return ret;
}


// [[Rcpp::export]]
Rcpp::List asSWMat(Rcpp::IntegerVector counts, int step, int nrow){
	Rcpp::List ret = Rcpp::List::create(Rcpp::Named("vec")=counts, Rcpp::Named("step")=step, Rcpp::Named("nrow")=nrow);
	ret.attr("class") = "swmat";
	return ret;
}


// [[Rcpp::export]]
double zScoreThresh(Rcpp::NumericVector lliks, double z, int nthreads=1){
	//compute mean and sd
	long double sum = 0;
	long double ssum = 0;
	double* it = lliks.begin();
	const int len = lliks.length();
	
	if (len<2) Rcpp::stop("vector too small to compute standard deviation");
	
	#pragma omp parallel for schedule(static) num_threads(nthreads) reduction(+: sum, ssum)
	for (int i = 0; i < len; ++i){
		double d = it[i];
		sum += d;
		ssum += d*d;
	}
	
	if (!std::isfinite(sum) || !std::isfinite(ssum)) return 0/0;//I couldn't figure out how to generate a NaN otherwise...
	
	double mean = sum/len;
	double sd = sqrt((ssum - mean*sum)/(len-1));
	
	return mean + z*sd;
}



// [[Rcpp::export]]
Rcpp::List fitModelFromColumns(SEXP gapmat, Rcpp::List model, Rcpp::List ucs, int negstrand = 0, int nthreads=1){
	GapMat<int> mat = asGapMat<int>(gapmat);
	
	if (negstrand < 0 || negstrand > mat.ncol) Rcpp::stop("invalid value for negstrand, must be between 0 and mat.ncol");
	int posstrand = mat.ncol - negstrand;
	
	double mu, r; double* ps; int footlen; parseModel(model, &mu, &r, &ps, &footlen);
	
	if (footlen != mat.nrow) Rcpp::stop("invalid model provided");
	
	Rcpp::IntegerVector uniqueCS = ucs["values"];
	Rcpp::IntegerVector map = ucs["map"];
	
	NMPreproc preproc(asVec(uniqueCS), asVec(map), Vec<double>(0,0));
	
	//we need an array of ones
	std::vector<double> std_ones(mat.ncol, 1); Mat<double> ones = asMat(std_ones, mat.ncol);
	//we need some temporary storage for fitting the negative binomial
	std::vector<double> std_tmpNB(uniqueCS.length());  Mat<double> tmpNB = asMat(std_tmpNB, 1);
	
	fitNBs_core(ones, Vec<double>(&mu, 1), Vec<double>(&r, 1), preproc, tmpNB, nthreads);
	
	
	//fitting multinomial
	Rcpp::NumericVector rcpp_newps(footlen); Vec<double> newps = asVec(rcpp_newps);
	
	
	std::vector<long double> tmpps(2*footlen);
	//positive strand
	if (posstrand > 0) rowSums(mat.subsetCol(0, posstrand), Vec<long double>(tmpps.data(), footlen), nthreads);
	//negative strand
	if (negstrand > 0) rowSums(mat.subsetCol(posstrand, mat.ncol), Vec<long double>(tmpps.data()+footlen, footlen), nthreads);
	//combining the two
	long double tot = 0;
	for (int i = 0; i < footlen; ++i){
		tmpps[i] += tmpps[2*footlen -i -1];
		tot += tmpps[i];
	}
	for (int i = 0; i < footlen; ++i){
		newps[i] = (tmpps[i]/tot);
	}

	return Rcpp::List::create(Rcpp::Named("mu")=mu, Rcpp::Named("r")=r, Rcpp::Named("ps")=rcpp_newps);
}


struct matIdx {
	int row;
	int col;
	
	matIdx(int _row, int _col): row(_row), col(_col) {}
};


// [[Rcpp::export]]
Rcpp::List filter(Rcpp::IntegerVector cols, Rcpp::NumericVector scores, const double thresh, int overlap, Rcpp::IntegerVector breaks){
	bool scanReverse = cols.length() != scores.length();
	if (scanReverse && 2*cols.length() != scores.length()) Rcpp::stop("non compatible dimensions");
	if (breaks.length() < 2 || breaks[0] != 0 || breaks[breaks.length()-1] != cols.length()) Rcpp::stop("invalid breaks vector provided");
	for (int i  = 1; i < breaks.length(); ++i){ if (breaks[i-1] > breaks[i]) Rcpp::stop("the breaks are not in ascending order"); }
	
	int len = cols.length();
	int nrow = scanReverse?2:1;
	
	if (overlap<=0){//just check where score > thresh
		int nchunks = breaks.length()-1;
		int vnum = nrow*nchunks;
		std::vector<int> std_starts(vnum);
		//it would have been cleaner to make them private variables to each thread,
		//but there is no resize() function for rcpp vector, so I need to use two
		//parallel regions because I don't know the size in advance.
		std::vector<std::vector<int> > idxs(nrow*nchunks);
		
		int* colsptr = cols.begin();

		Mat<int> starts = asMat(std_starts, nchunks);
		Mat<std::vector<int> > idxsMat = asMat(idxs, nchunks);
		
		
		Mat<double> scoresMat = Mat<double>(scores.begin(), nrow, len);
		int reverse = 0;
		
		//accumulate valid indices in the appropriate vector according to the row and the thread
		#pragma omp parallel for num_threads(nchunks)
		for (int t = 0; t < nchunks; ++t){
			Vec<std::vector<int> > idxsCol = idxsMat.getCol(t);
			int is = breaks[t], ie = breaks[t+1];
			double* scoresptr = scoresMat.colptr(is);
			for (int i = is*nrow, e = ie*nrow; i < e; ++i, ++scoresptr){
				if (*scoresptr > thresh){
				//you should implement a custom resize strategy that takes
				// into account how many indices are left and the growing rate
					idxsCol[i%nrow].push_back(colsptr[i/nrow]);
				}
			}
		}
		
		//std::cout << "accumulation done" << std::endl;
		
		//check how much memory needs to be allocated and decide in which portion of the final array 
		//each vector will be copied 
		int totlen = 0;
		for (int row = 0; row < nrow; ++row){
			for (int t = 0; t < nchunks; ++t){
				starts(row, t) = totlen;
				totlen += idxsMat(row, t).size();
			}
		}
		Rcpp::IntegerVector res(totlen);
		if (nrow > 1) reverse = totlen - starts(1, 0);
		
		//each thread writes its private memory to the final array
		#pragma omp parallel for num_threads(nchunks)
		for (int t = 0; t < nchunks; ++t){
			for (int j = 0; j < nrow; ++j){
				memcpy(res.begin() + starts(j, t), idxsMat(j, t).data(), sizeof(int)*idxsMat(j,t).size());
			}
		}
		
		return Rcpp::List::create(Rcpp::Named("idxs")=res, Rcpp::Named("reverse")=reverse);
	} else {
		/* forward backward-like algorithm for removing overlaps (no sorting) */
		//cols are assumed to be already sorted!!!!!!
		std::vector<matIdx> idxs;
		int* colsptr = cols.begin();
		double* scoresptr = scores.begin();
		
		int lastCol = colsptr[0] - overlap; //nobody will ever overlap with this position
		double lastScore = thresh;
		
			
		//forward
		for (int i = 0, e = nrow*len; i < e; ++i){
			if (*scoresptr > thresh){
				int col = colsptr[i/nrow]; 
				double score = *scoresptr;
				if ((col - lastCol >= overlap) || (score > lastScore)){ //valid position: if it overlaps, the score has to be better than the last score
					idxs.push_back(matIdx(i%nrow, col));
					lastScore = score;
					lastCol = col;
				}
			}
			++scoresptr;
		}
		//backward, here we don't resize the array, we just signal with a row=-1 when an element is removed
		int tot = 0; int neg = 0;
		if (idxs.size() > 0){
			lastCol = idxs[idxs.size()-1].col + overlap; //nobody will ever overlap with this position
			for (int i = idxs.size() - 1; i >= 0; --i){
				if (lastCol - idxs[i].col >= overlap){ //valid position: it doesn't overlap with the next interval
					lastCol = idxs[i].col;
					neg += idxs[i].row;
					++tot;
				} else {//overlaps, must be excluded
					idxs[i].row = -1;
				}
			}
		}
		//fill the final array
		Rcpp::IntegerVector rcppidxs(tot);
		int* posidx = rcppidxs.begin(); int* negidx = posidx + tot - neg;
		for (int i = 0; i < idxs.size(); ++i){
			switch (idxs[i].row){
				case 0: *posidx = idxs[i].col; ++posidx; break;
				case 1: *negidx = idxs[i].col; ++negidx; break;
			}
		}
		return Rcpp::List::create(Rcpp::Named("idxs")=rcppidxs, Rcpp::Named("reverse")=neg);
	}
}

// [[Rcpp::export]]
Rcpp::IntegerVector removeOverlapping(Rcpp::IntegerVector cols, Rcpp::IntegerVector centers, int radius){
	if (radius <= 0) Rcpp::stop("null or negative radii don't make sense bro...");
	if (cols.length()*centers.length() == 0) return cols;
	
	//get the starts from the centers, let's also add a fake start, higher than the last column
	std::vector<int> starts(centers.length() + 1); int* s = starts.data(); int* c = centers.begin();
	for (int i = 0, e = centers.length(); i < e; ++i){ *s = *c - radius + 1; ++s; ++c;}
	*s = cols[cols.length()-1] + 1;//fake start, no column will ever overlap with this interval
	int width = 2*radius - 1;
	
	//sort the starts
	std::sort(starts.begin(), starts.end());
	//go on with filtering. cols are assumed to be already sorted
	std::vector<int> filtered;
	c = cols.begin();  //pointers to the last index. 
	s = starts.data(); //pointer to the last interval
	int* c_end = cols.end(); int* s_end = s + starts.size();
	
	for (; s < s_end; ++s){
		int stop = *s;
		while (c < c_end && *c < stop){//these guys do not overlap neither with the previous interval nor with the current
			filtered.push_back(*c); ++c;
		}
		stop = *s + width;
		while (c < c_end && *c < stop){ ++c; }//these guys overlap with the interval [*s, *s+width]
	}
	
	Rcpp::IntegerVector res = Rcpp::wrap(filtered);
	return res;
}


// [[Rcpp::export]]
void nbinom_llik(double mu, double r, Rcpp::IntegerVector uniqueCS, Rcpp::NumericVector tmpNB, int nthreads){
	lLik_nbinom(mu, r, asVec(uniqueCS), asVec(tmpNB), nthreads);
}

// [[Rcpp::export]]
void multinom_llik(SEXP gapmat, Rcpp::NumericVector ps, Rcpp::NumericVector llik, Rcpp::IntegerVector map, Rcpp::NumericVector tmpNB, Rcpp::NumericVector  mconst, int nthreads){
	lLik_multinom(asGapMat<int>(gapmat), asVec(ps), asVec(llik), asVec(map), asVec(tmpNB), asVec(mconst), nthreads);
}



// [[Rcpp::export]]
Rcpp::IntegerVector findBreaks(Rcpp::IntegerVector colset, int overlap, int nthreads){
	int len = colset.length();
	double dn = nthreads;
	
	if (nthreads == 1){ Rcpp::IntegerVector res(2); res[0] = 0; res[1] = len; return res; }
	
	if (overlap <= 0){
		Rcpp::IntegerVector res(nthreads+1);
		for (int i = 1; i <= nthreads; ++i){
			res[i] = round(len*(i/dn));
		}
		return res;
	} else {
		std::vector<int> breaks(nthreads+1);
		int valid = 0;
		
		#pragma omp parallel for num_threads(nthreads-1) reduction(+:valid)
		for (int i = 1; i < nthreads; ++i){
			//just greedily choosing the break point closest to the center
			int start = std::max(1, round(len*((i-0.5)/dn)));
			int center = round(len*(i/dn));
			int end = round(len*((i+0.5)/dn));
			//I want to make sure that:
			// start_i <= center_i <= end_i (apart from cases when len is very small)
			// start_(i+1) = end_i (the whole range is covered: if there is only one break, you will find it)
			// start_i <= break_i < end_i 
			// break_i < break_(i+1) (no break is repeated)
			
			//std::cout << std::string("start: ") + itoa(start) + " end: " + itoa(end) + "\n" + << std::endl;
			
			int* coords = colset.begin();
			int bpoint = -1;
			for (int j = std::min(center, end-1); j >= start; --j){
				if (coords[j]-coords[j-1] >= overlap) {
					bpoint = j; break;
				}
			}
			int newend = end;
			if (bpoint != -1){ newend = std::min(end, 2*center - bpoint); }
			for (int j = center+1; j < newend; ++j){
				if (coords[j]-coords[j-1] >= overlap) {
					bpoint = j; break;
				}
			}
			breaks[i] = bpoint;
			
			//printf("start: %i, end: %i, center: %i, bpoint: %i\n", start, end, center, bpoint);
			
			
			if (breaks[i] >= 0) valid = 1;
		}
		
		breaks[0] = 0;
		breaks[nthreads] = len;
		valid += 2;
		
		Rcpp::IntegerVector res(valid); int j = 0;
		for (int i = 0; i <= nthreads; ++i){
			if (breaks[i] >= 0) res[j++] = breaks[i];
		}
		
		return res;
	}
}
