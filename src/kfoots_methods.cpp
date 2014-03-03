#include <Rcpp.h>
#include "core.hpp"
#include <sys/time.h>
#include <algorithm> 

// [[Rcpp::export]]
Rcpp::List llik2posteriors(Rcpp::NumericMatrix lliks, Rcpp::NumericVector mix_coeff, Rcpp::NumericMatrix posteriors, int nthreads=1){
	//copy the vector (I hope...)
	Rcpp::NumericVector new_mix_coeff(mix_coeff);
	
	double tot = llik2posteriors_core(asMat(lliks), asVec(new_mix_coeff), asMat(posteriors), nthreads);
	
	return Rcpp::List::create(
									Rcpp::Named("posteriors")=posteriors,
									Rcpp::Named("tot_llik")=tot,
									Rcpp::Named("new_mix_coeff")=new_mix_coeff);
}


//' Group unique values of a vector
//'
//' @param v a vector of integers. If they are not integers they will be
//' 	casted to integers.
//' @return a list with the following items:
//'		\item{values}{unique and sorted values of \code{v}}
//'		\item{map}{a vector such that \code{v[i] = values[map[i]+1]} for every i}
//'	@export
// [[Rcpp::export]]
Rcpp::List mapToUnique(Rcpp::IntegerVector values){
	Rcpp::IntegerVector map(values.length());
	
	Vec<int> valuesVec = asVec(values);
	Vec<int> mapVec = asVec(map);
	std::vector<int> uniqueCS;
	map2unique_core(valuesVec, mapVec, uniqueCS);
	
	return Rcpp::List::create(Rcpp::Named("values")=Rcpp::IntegerVector(uniqueCS.begin(),uniqueCS.end()), Rcpp::Named("map")=map);
}


// [[Rcpp::export]]
Rcpp::List subsetM2U(Rcpp::List ucs, Rcpp::IntegerVector colidxs){
	Rcpp::IntegerVector map = Rcpp::as<Rcpp::IntegerVector>(ucs["map"]);
	Rcpp::IntegerVector values = Rcpp::as<Rcpp::IntegerVector>(ucs["values"]);
	Rcpp::IntegerVector newmap(colidxs.length());
	std::vector<int> newvalues;
	subsetM2U_core(asVec(values), asVec(map), asVec(colidxs), newvalues, asVec(newmap));
	return Rcpp::List::create(Rcpp::Named("values")=Rcpp::IntegerVector(newvalues.begin(),newvalues.end()), Rcpp::Named("map")=newmap);
}


template<template <typename> class TMat>
inline Rcpp::NumericVector getMultinomConst_helper(TMat<int> counts, int nthreads=1){
	Rcpp::NumericVector multinomConst(counts.ncol);
	getMultinomConst_core(counts, asVec(multinomConst), nthreads);
	return multinomConst;
}

// [[Rcpp::export]]
Rcpp::NumericVector getMultinomConst(Rcpp::IntegerMatrix counts, int nthreads=1){ return getMultinomConst_helper(asMat(counts), nthreads);}

// [[Rcpp::export]]
Rcpp::NumericVector getMultinomConstSW(SEXP counts, int nthreads=1){	return getMultinomConst_helper(asSWMat<int>(counts), nthreads);}


// [[Rcpp::export]]
Rcpp::NumericVector sumAt(Rcpp::NumericVector values, Rcpp::IntegerVector map, int size, bool zeroIdx=false){
	typedef Rcpp::NumericVector::iterator diter;
	typedef Rcpp::IntegerVector::iterator iiter;

	Rcpp::NumericVector res(size);
	diter vend = values.end();
	diter vstart = values.begin();
	iiter mstart = map.begin();
	
	if (zeroIdx){
		for (; vstart!=vend; ++vstart, ++mstart){
			res[(*mstart)] += *vstart;
		}
	} else {
		for (; vstart!=vend; ++vstart, ++mstart){
			res[(*mstart)-1] += *vstart;
		}

	}
	return res;
}

template<template <typename> class TMat>
inline Rcpp::IntegerVector colSumsInt_helper(TMat<int> mat, int nthreads=1){
	Rcpp::IntegerVector ret(mat.ncol);
	colSums(mat, asVec(ret), nthreads);
	return ret;
}

// [[Rcpp::export]]
Rcpp::IntegerVector colSumsInt(Rcpp::IntegerMatrix nums, int nthreads=1){return colSumsInt_helper(asMat(nums), nthreads);}

// [[Rcpp::export]]
Rcpp::IntegerVector colSumsIntSW(SEXP nums, int nthreads=1){return colSumsInt_helper(asSWMat<int>(nums), nthreads);}


// [[Rcpp::export]]
Rcpp::NumericVector colSumsDouble(Rcpp::NumericMatrix nums, int nthreads=1){
	Mat<double> mat = asMat(nums);
	Rcpp::NumericVector ret(mat.ncol);
	Vec<double> vec = asVec(ret);
	colSums(mat, vec, nthreads);
	return ret;
}

// [[Rcpp::export]]
Rcpp::NumericVector rowSumsDouble(Rcpp::NumericMatrix mat, int nthreads=1){
	std::vector<long double> acc(mat.nrow(), 0);
	rowSums<double, long double>(asMat(mat), asVec(acc), nthreads);
	Rcpp::NumericVector ret(mat.nrow());
	for (int i = 0, e = mat.nrow(); i < e; ++i){ret[i] = acc[i];}
	return ret;
}

/*
// [[Rcpp::export]]
Rcpp::NumericVector nbinomLoglik(Rcpp::IntegerVector counts, double mu, double r, int nthreads=1){
	Rcpp::NumericVector res(counts.length());
	nbinomLoglik_core(asVec(counts), mu, r, asVec(res), std::max(nthreads, 1));
	return res;
}
*/

/*
// [[Rcpp::export]]
Rcpp::NumericVector fitMultinom(Rcpp::IntegerMatrix counts, Rcpp::NumericVector posteriors, int nthreads=1){
	Mat<int> mat = asMat(counts);
	Rcpp::NumericVector fit(mat.nrow);
	fitMultinom_core(mat, asVec(posteriors), asVec(fit), std::max(nthreads, 1));
	return fit;
}
*/

static inline Rcpp::List writeModels(Vec<double> mus, Vec<double> rs, Mat<double> ps){
	int nmod = mus.len;
	unsigned int footsize = ps.nrow;
	Rcpp::List ret(nmod);
	for (int i = 0; i < nmod; ++i){
		Rcpp::NumericVector currps(footsize);
		memcpy(currps.begin(), ps.colptr(i), footsize*sizeof(double));
		ret[i] = Rcpp::List::create(Rcpp::Named("mu")=mus[i], Rcpp::Named("r")=rs[i], Rcpp::Named("ps")=currps);
	}
	return ret;
}


/*
// [[Rcpp::export]]
Rcpp::NumericVector lLik(Rcpp::IntegerMatrix counts, Rcpp::List model, Rcpp::List ucs, Rcpp::NumericVector mConst, int nthreads=1){
	
	if (ucs.length()==0){
		ucs = mapToUnique(colSumsInt(counts, nthreads));
	}
	if (mConst.length()==0){
		mConst = getMultinomConst(counts, nthreads);
	}
	
	Rcpp::IntegerVector map = ucs["map"];
	Rcpp::IntegerVector uniqueCS = ucs["values"];
	NMPreproc preproc(asVec(uniqueCS), asVec(map), asVec(mConst));
	
	Mat<int> countsMat = asMat(counts);
	Rcpp::NumericVector lliks(countsMat.ncol);
	Vec<double> lliksVec = asVec(lliks);
	double mu, r; double* ps; int footlen;
	parseModel(model, &mu, &r, &ps, &footlen);
	
	//re-format preprocessing data if present, otherwise, create it.
	//If created here they will not be persistent
	
	getLlik(countsMat, mu, r, Vec<double>(ps, footlen), lliksVec, preproc, nthreads);
	
	return lliks;
}
*/


template<template <typename> class TMat>
inline void lLikMat_helper(TMat<int> counts, Rcpp::List models, 
		Rcpp::List ucs, Rcpp::NumericVector mConst, Rcpp::NumericVector lliks,
		int nthreads=1){
	
	
	//parse or compute preprocessing data
	if (ucs.length()==0){
		ucs = mapToUnique(colSumsInt_helper(counts, nthreads));
	}
	if (mConst.length()==0){
		mConst = getMultinomConst_helper(counts, nthreads);
	}
	
	Rcpp::IntegerVector uniqueCS = Rcpp::as<Rcpp::IntegerVector>(ucs["values"]);
	Rcpp::IntegerVector map = Rcpp::as<Rcpp::IntegerVector>(ucs["map"]);
	NMPreproc preproc(asVec(uniqueCS), asVec(map), asVec(mConst));
	
	int nmodels = models.length();
	int footlen = counts.nrow;
	//parsing the models
	std::vector<double> musSTD(nmodels); Vec<double> mus = asVec(musSTD);
	std::vector<double> rsSTD(nmodels); Vec<double> rs = asVec(rsSTD);
	std::vector<double> psSTD(nmodels*footlen); Mat<double> ps = asMat(psSTD, nmodels);
	parseModels(models, mus, rs, ps);
	//re-formatting the lliks vector
	if (models.length()*counts.ncol != lliks.length()) Rcpp::stop("wrong length for the lliks vector");
	Mat<double> lliksmat(lliks.begin(), nmodels, counts.ncol);
	
	//allocating some temporary memory
	std::vector<double> tmpNB(uniqueCS.length()*nmodels);
	
	lLikMat_core(counts, mus, rs, ps, lliksmat, preproc, asMat(tmpNB, uniqueCS.length()), nthreads);
}

// [[Rcpp::export]]
void lLikMat(	Rcpp::IntegerMatrix counts, Rcpp::List models, 
					Rcpp::List ucs, Rcpp::NumericVector mConst, Rcpp::NumericVector lliks,
					int nthreads=1){
	lLikMat_helper(asMat(counts), models, ucs, mConst, lliks, nthreads);
}

// [[Rcpp::export]]
void lLikGapMat(	SEXP counts, Rcpp::List models, Rcpp::List ucs,
						Rcpp::NumericVector mConst, Rcpp::NumericVector lliks,
						int nthreads=1){
	lLikMat_helper(asGapMat<int>(counts), models, ucs, mConst, lliks, nthreads);
}


// [[Rcpp::export]]
Rcpp::IntegerVector pwhichmax(Rcpp::NumericMatrix posteriors, int nthreads=1){
	Rcpp::IntegerVector clusters(posteriors.ncol());
	pwhichmax_core(asMat(posteriors), asVec(clusters), nthreads);
	return clusters;
}

// [[Rcpp::export]]
Rcpp::List fitNB_inner(Rcpp::IntegerVector counts, Rcpp::NumericVector posteriors, double initR=-1){
	double mu = -1;
	double r = -1;
	fitNB_core(asVec(counts), asVec(posteriors), &mu, &r, initR);
	
	return Rcpp::List::create(Rcpp::Named("mu")=mu, Rcpp::Named("r")=r);
}


template<template <typename> class TMat>
inline Rcpp::List fitModels_helper(TMat<int> counts, Rcpp::NumericVector posteriors, Rcpp::List models, Rcpp::List ucs, int nthreads=1){
	int nmodels = models.length();
	int footlen = counts.nrow;
	
	if (	counts.ncol*nmodels != posteriors.length()){
		Rcpp::stop("Invalid arguments passed to fitModels");
	}
	Mat<double> postMat(posteriors.begin(), nmodels, counts.ncol);
	
	//parse or compute preprocessing data (multinomConst is not needed)
	if (ucs.length()==0){
		ucs = mapToUnique(colSumsInt_helper(counts, nthreads));
	}
	
	Rcpp::IntegerVector uniqueCS = ucs["values"];
	Rcpp::IntegerVector map = ucs["map"];

	NMPreproc preproc(asVec(uniqueCS), asVec(map), Vec<double>(0,0));
	
	//parsing the models, this memory will also store the new parameters
	std::vector<double> musSTD(nmodels); Vec<double> mus = asVec(musSTD);
	std::vector<double> rsSTD(nmodels); Vec<double> rs = asVec(rsSTD);
	std::vector<double> psSTD(nmodels*footlen); Mat<double> ps = asMat(psSTD, nmodels);
	parseModels(models, Vec<double>(0,0), rs, Mat<double>(0,0,0));//we only care about the rs
	
	//allocating some temporary memory
	std::vector<double> tmpNB(uniqueCS.length()*nmodels);
	
	fitNBs_core(postMat, mus, rs, preproc, asMat(tmpNB, nmodels), nthreads);
	fitMultinoms_core(counts, postMat, ps, nthreads);
	return writeModels(mus, rs, ps);
}

// [[Rcpp::export]]
Rcpp::List fitModels(Rcpp::IntegerMatrix counts, Rcpp::NumericMatrix posteriors, Rcpp::List models, Rcpp::List ucs, int nthreads=1){
	return fitModels_helper(asMat(counts), posteriors, models, ucs, nthreads);
}

// [[Rcpp::export]]
Rcpp::List fitModelsGapMat(SEXP counts, Rcpp::NumericMatrix posteriors, Rcpp::List models, Rcpp::List ucs, int nthreads=1){
	return fitModels_helper(asGapMat<int>(counts), posteriors, models, ucs, nthreads);
}


/*
// [[Rcpp::export]]
void fitNBs(
						Rcpp::NumericMatrix post, 
						Rcpp::NumericVector mus, 
						Rcpp::NumericVector rs,
						Rcpp::List ucs,
						int nthreads = 1){
	
	Rcpp::IntegerVector uniqueCS = ucs["values"];
	Rcpp::IntegerVector map = ucs["map"];
	NMPreproc preproc(asVec(uniqueCS), asVec(map), Vec<double>(0,0));
	std::vector<double> tmpNB(uniqueCS.length()*mus.length());
	fitNBs_core(asMat(post), asVec(mus), asVec(rs), preproc, asMat(tmpNB, mus.length()), nthreads);
}
*/

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
	
	//parse or compute preprocessing data (multinomConst is not needed)
	if (ucs.length()==0){
		ucs = mapToUnique(colSumsInt_helper(mat, nthreads));
	}
	
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
Rcpp::List filter(Rcpp::IntegerVector cols, Rcpp::NumericVector scores, const double thresh, int overlap, Rcpp::IntegerVector breaks, int nthreads){
	bool scanReverse = cols.length() != scores.length();
	if (scanReverse && 2*cols.length() != scores.length()) Rcpp::stop("non compatible dimensions");
	
	int len = cols.length();
	int nrow = scanReverse?2:1;
	
	if (overlap<=0){//just check where score > thresh
		std::vector<int> idxs;
		int nonreverse = 0;
		int* colsptr = cols.begin();
		
		//loop on the rows
		for (int j = 0; j < nrow; ++j){
			double* scoresptr = scores.begin() + j;
			for (int i = 0; i < len; ++i){
				if (*scoresptr > thresh){
					idxs.push_back(colsptr[i]);
				}
				scoresptr += nrow;
			}
			if (j == 0){ nonreverse = idxs.size(); }
		}
		//std::cout << "idxs len: " << idxs.size() << std::endl;
		return Rcpp::List::create(Rcpp::Named("idxs")=Rcpp::wrap(idxs), Rcpp::Named("reverse")=(idxs.size()-nonreverse));
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
			int start = round(len*((i-0.5)/dn));
			int center = round(len*(i/dn));
			int end = round(len*((i+0.5)/dn));
			int bestBreak = -1;
			int* coords = colset.begin();
			for (int j = start+1; j <= center; ++j){
				if (coords[j]-coords[j-1] >= overlap) bestBreak = j;
			}
			for (int j = end; j > center; --j){
				if (coords[j]-coords[j-1] >= overlap) bestBreak = j;
			}
			breaks[i] = bestBreak;
			
			if (bestBreak >= 0) valid = 1;
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
