#include <Rcpp.h>
#include <boost/unordered_map.hpp>
#include "getLlik.hpp"

// [[Rcpp::export]]
Rcpp::List llik2posteriors(Rcpp::NumericMatrix lliks, int nthreads=1){
	Rcpp::NumericMatrix posteriors(lliks.nrow(), lliks.ncol());
	double tot = llik2posteriors_core(asMat<double>(lliks), asMat<double>(posteriors), nthreads);
	
	return Rcpp::List::create(Rcpp::Named("posteriors")=posteriors, Rcpp::Named("tot_llik")=tot);
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
	Vec<int> valuesVec = asVec<int>(values);
	Vec<int> mapVec = asVec<int>(map);
	std::vector<int> uniqueCS;
	map2unique_core(valuesVec, mapVec, uniqueCS);
	return Rcpp::List::create(Rcpp::Named("values")=Rcpp::IntegerVector(uniqueCS.begin(),uniqueCS.end()), Rcpp::Named("map")=map);
}



// [[Rcpp::export]]
Rcpp::NumericVector getMultinomConst(Rcpp::IntegerMatrix counts, int nthreads=1){
	Rcpp::NumericVector multinomConst(counts.ncol());
	getMultinomConst_core(asMat<int>(counts), asVec<double>(multinomConst), nthreads);
	return multinomConst;
}

typedef Rcpp::NumericVector::iterator diter;
typedef Rcpp::IntegerVector::iterator iiter;
// [[Rcpp::export]]
Rcpp::NumericVector sumAt(Rcpp::NumericVector values, Rcpp::IntegerVector map, int size, bool zeroIdx=false){
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



// [[Rcpp::export]]
Rcpp::NumericVector lLik_inner(Rcpp::IntegerMatrix counts, Rcpp::List model, 
		Rcpp::IntegerVector uniqueCS,
		Rcpp::IntegerVector map,
		Rcpp::NumericVector multinomConst,
		int nthreads=1){
	
	Mat<int> countsMat = asMat<int>(counts);
	Rcpp::NumericVector lliks(counts.ncol());
	Vec<double> lliksVec = asVec<double>(lliks);
	Rcpp::NumericVector ps = model["ps"];
	NegMultinom NMmodel(model["r"], model["mu"], Vec<double>(ps.begin(), ps.length()));
	
	//re-format preprocessing data if present, otherwise, create it.
	//If created here they will not be persistent
	NMPreproc preproc(asVec<int>(uniqueCS), asVec<int>(map), asVec<double>(multinomConst));
	
	NMmodel.getLlik(countsMat, lliksVec, preproc, nthreads);
	
	return lliks;
}

/*
// [[Rcpp::export]]
Rcpp::List tryPreproc(Rcpp::IntegerMatrix counts, Rcpp::List model){
	Mat<int> countsMat = asMat<int>(counts);
	NMPreproc preproc(countsMat);
	
	Rcpp::NumericVector ps = model["ps"];
	NegMultinom NMmodel(model["r"], model["mu"], Vec<double>(ps.begin(), ps.length()));
	std::vector<double> nbinoms(preproc.uniqueCS.size());
	for (int i = 0; i < nbinoms.size(); ++i){
		nbinoms[i] = NMmodel.logNBinom(preproc.uniqueCS[i]);
	}
	
	return Rcpp::List::create(
		Rcpp::Named("uniqueCS")=preproc.uniqueCS,
		Rcpp::Named("map")=preproc.map,
		Rcpp::Named("multinomConst")=preproc.multinomConst
		//Rcpp::Named("lognbinoms")=nbinoms
		);
}
*/


// [[Rcpp::export]]
Rcpp::NumericVector lfactorial2(Rcpp::IntegerVector nums){
	Rcpp::NumericVector ret(nums.length());
	double* d = ret.begin();
	const int* e = nums.end();
	
	CachedLFact lfact(0.75);
	
	for (int* i = nums.begin(); i < e; ++i, ++d){
		*d = lfact(*i);
		
	}
	
	return ret;
}



// [[Rcpp::export]]
Rcpp::IntegerVector colSumsInt(Rcpp::IntegerMatrix nums, int nthreads=1){
	Mat<int> mat = asMat<int>(nums);
	Rcpp::IntegerVector ret(mat.ncol);
	Vec<int> vec = asVec<int>(ret);
	
	colSums(mat, vec, nthreads);
	return ret;
}

// [[Rcpp::export]]
Rcpp::NumericVector colSumsDouble(Rcpp::NumericMatrix nums, int nthreads=1){
	Mat<double> mat = asMat<double>(nums);
	Rcpp::NumericVector ret(mat.ncol);
	Vec<double> vec = asVec<double>(ret);
	
	colSums(mat, vec, nthreads);
	return ret;
}

// [[Rcpp::export]]
Rcpp::NumericVector nbinomLoglik(Rcpp::IntegerVector counts, double mu, double r, int nthreads=1){
	Rcpp::NumericVector res(counts.length());
	nbinomLoglik_core(asVec<int>(counts), mu, r, asVec<double>(res), std::max(nthreads, 1));
	return res;
}

// [[Rcpp::export]]
double optimFun(Rcpp::IntegerVector counts, double mu, double r, Rcpp::NumericVector posteriors, int nthreads=1){
	return optimFun_core(asVec<int>(counts), mu, r, asVec<double>(posteriors), std::max(nthreads, 1));
}

// [[Rcpp::export]]
Rcpp::NumericVector fitMultinom(Rcpp::IntegerMatrix counts, Rcpp::NumericVector posteriors, int nthreads=1){
	Rcpp::NumericVector fit(counts.nrow());
	fitMultinom_core(asMat<int>(counts), asVec<double>(posteriors), asVec<double>(fit), std::max(nthreads, 1));
	return fit;
}
