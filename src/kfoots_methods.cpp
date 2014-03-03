#include <Rcpp.h>
#include "core.hpp"
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
