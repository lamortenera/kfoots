#include <Rcpp.h>
#include <boost/unordered_map.hpp>
#include "core.hpp"
#include <sys/time.h>

// [[Rcpp::export]]
Rcpp::List llik2posteriors(Rcpp::NumericMatrix lliks, Rcpp::NumericVector mix_coeff, SEXP posteriors=R_NilValue, int nthreads=1){
	if (Rf_isNull(posteriors)){
		posteriors = Rcpp::NumericMatrix(lliks.nrow(), lliks.ncol());
	}
	
	Rcpp::NumericMatrix tposteriors(posteriors);
	//copy the vector (I hope...)
	Rcpp::NumericVector new_mix_coeff(mix_coeff);
	
	double tot = llik2posteriors_core(asMat(lliks), asVec(new_mix_coeff), asMat(tposteriors), nthreads);
	
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


// [[Rcpp::export]]
Rcpp::NumericVector getMultinomConst(Rcpp::RObject counts, int nthreads=1){
	MatWrapper<int> mat = wrapMat<INTSXP>(counts);
	Rcpp::NumericVector multinomConst(mat.ncol);
	if (mat.type == "matrix"){
		getMultinomConst_core(mat.matrix, asVec(multinomConst), nthreads);
	} else if (mat.type == "gapmat"){
		getMultinomConst_core(mat.gapmat, asVec(multinomConst), nthreads);
	} else {
		getMultinomConst_core(mat.swmat, asVec(multinomConst), nthreads);
	}
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
Rcpp::IntegerVector colSumsInt(Rcpp::RObject nums, int nthreads=1){
	MatWrapper<int> mat = wrapMat<INTSXP>(nums);
	Rcpp::IntegerVector ret(mat.ncol);
	Vec<int> vec = asVec(ret);
	if (mat.type == "matrix"){
		colSums(mat.matrix, vec, nthreads);
	} else if (mat.type == "gapmat"){
		colSums(mat.gapmat, vec, nthreads);
	} else {
		colSums(mat.swmat, vec, nthreads);
	}
	return ret;
}

// [[Rcpp::export]]
Rcpp::NumericVector colSumsDouble(Rcpp::NumericMatrix nums, int nthreads=1){
	Mat<double> mat = asMat(nums);
	Rcpp::NumericVector ret(mat.ncol);
	Vec<double> vec = asVec(ret);
	
	colSums(mat, vec, nthreads);
	return ret;
}

// [[Rcpp::export]]
Rcpp::NumericVector nbinomLoglik(Rcpp::IntegerVector counts, double mu, double r, int nthreads=1){
	Rcpp::NumericVector res(counts.length());
	nbinomLoglik_core(asVec(counts), mu, r, asVec(res), std::max(nthreads, 1));
	return res;
}

// [[Rcpp::export]]
double optimFun(Rcpp::IntegerVector counts, double mu, double r, Rcpp::NumericVector posteriors, int nthreads=1){
	return optimFun_core(asVec(counts), mu, r, asVec(posteriors), std::max(nthreads, 1));
}

// [[Rcpp::export]]
Rcpp::NumericVector fitMultinom(Rcpp::RObject counts, Rcpp::NumericVector posteriors, int nthreads=1){
	MatWrapper<int> mat = wrapMat<INTSXP>(counts);
	Rcpp::NumericVector fit(mat.nrow);
	if (mat.type=="matrix"){
		fitMultinom_core(mat.matrix, asVec(posteriors), asVec(fit), std::max(nthreads, 1));
	} else if (mat.type=="gapmat"){
		fitMultinom_core(mat.gapmat, asVec(posteriors), asVec(fit), std::max(nthreads, 1));
	} else {
		fitMultinom_core(mat.swmat, asVec(posteriors), asVec(fit), std::max(nthreads, 1));
	}
	return fit;
}

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

// [[Rcpp::export]]
Rcpp::NumericVector lLik(Rcpp::RObject counts, Rcpp::List model, 
		SEXP ucs = R_NilValue,
		SEXP mConst = R_NilValue,
		int nthreads=1){
	
	if (Rf_isNull(ucs)){
		ucs = Rcpp::wrap(mapToUnique(colSumsInt(counts, nthreads)));
	}
	if (Rf_isNull(mConst)){
		mConst = Rcpp::wrap(getMultinomConst(counts, nthreads));
	}
	
	Rcpp::List ucs_list(ucs); 
	Rcpp::IntegerVector map = ucs_list["map"];
	Rcpp::IntegerVector uniqueCS = ucs_list["values"];
	Rcpp::NumericVector multinomConst(mConst);
	
	MatWrapper<int> countsMat = wrapMat<INTSXP>(counts);
	Rcpp::NumericVector lliks(countsMat.ncol);
	Vec<double> lliksVec = asVec(lliks);
	double mu, r; double* ps; int footlen;
	parseModel(model, &mu, &r, &ps, &footlen);
	
	//re-format preprocessing data if present, otherwise, create it.
	//If created here they will not be persistent
	NMPreproc preproc(asVec(uniqueCS), asVec(map), asVec(multinomConst));
	
	if (countsMat.type == "matrix"){
		getLlik(countsMat.matrix, mu, r, Vec<double>(ps, footlen), lliksVec, preproc, nthreads);
	} else if (countsMat.type == "gapmat"){
		getLlik(countsMat.gapmat, mu, r, Vec<double>(ps, footlen), lliksVec, preproc, nthreads);
	} else {
		getLlik(countsMat.swmat, mu, r, Vec<double>(ps, footlen), lliksVec, preproc, nthreads);
	}
	
	return lliks;
}


// [[Rcpp::export]]
Rcpp::NumericMatrix lLikMat(Rcpp::IntegerMatrix counts, Rcpp::List models, 
		SEXP ucs = R_NilValue,
		SEXP mConst = R_NilValue,
		SEXP lliks = R_NilValue,
		int nthreads=1){
	
	//parse or compute preprocessing data
	if (Rf_isNull(ucs)){
		ucs = (SEXP) mapToUnique(colSumsInt(counts, nthreads));
	}
	if (Rf_isNull(mConst)){
		mConst = (SEXP) getMultinomConst(counts, nthreads);
	}
	
	Rcpp::List ucs_list(ucs); 
	Rcpp::IntegerVector uniqueCS = ucs_list["values"];
	Rcpp::IntegerVector map = ucs_list["map"];
	Rcpp::NumericVector multinomConst(mConst);
	NMPreproc preproc(asVec(uniqueCS), asVec(map), asVec(multinomConst));
	
	Mat<int> countsMat = asMat(counts);
	//parsing the models
	int nmodels = models.length();
	int footlen = countsMat.nrow;
	std::vector<double> musSTD(nmodels);
	std::vector<double> rsSTD(nmodels);
	std::vector<double> psSTD(nmodels*footlen);
	Vec<double> mus = asVec(musSTD);
	Vec<double> rs = asVec(rsSTD);
	Mat<double> ps = asMat(psSTD, nmodels);
	parseModels(models, mus, rs, ps);
	//allocating some temporary memory
	std::vector<double> tmpNB(uniqueCS.length()*nmodels);
	//allocating return variable
	if (Rf_isNull(lliks)){
		lliks = Rcpp::NumericMatrix(nmodels, countsMat.ncol);
	}
	Rcpp::NumericMatrix tlliks(lliks);
	
	lLikMat_core(countsMat, mus, rs, ps, asMat(tlliks), preproc, asMat(tmpNB, uniqueCS.length()), nthreads);
	
	return lliks;
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

// [[Rcpp::export]]
Rcpp::List fitModels(Rcpp::IntegerMatrix counts, Rcpp::NumericMatrix posteriors, Rcpp::List models, 
	SEXP ucs = R_NilValue,
	int nthreads=1){
	
	int nmodels = models.length();
	int footlen = counts.nrow();
	
	if (	counts.ncol() != posteriors.ncol() ||
			posteriors.nrow() != nmodels ){
		throw std::invalid_argument("Invalid arguments passed to fitModels");
	}
	
	//parse or compute preprocessing data (multinomConst is not needed)
	if (Rf_isNull(ucs)){
		ucs = (SEXP) mapToUnique(colSumsInt(counts, nthreads));
	}
	Rcpp::List ucs_list(ucs); 
	Rcpp::IntegerVector uniqueCS = ucs_list["values"];
	Rcpp::IntegerVector map = ucs_list["map"];
	NMPreproc preproc(asVec(uniqueCS), asVec(map), Vec<double>(0,0));
	
	Mat<int> countsMat = asMat(counts);
	Mat<double> postMat = asMat(posteriors);
	//parsing the models
	std::vector<double> musSTD(nmodels);
	std::vector<double> rsSTD(nmodels);
	std::vector<double> psSTD(nmodels*footlen);
	Vec<double> mus = asVec(musSTD);
	Vec<double> rs = asVec(rsSTD);
	Mat<double> ps = asMat(psSTD, nmodels);
	parseModels(models, Vec<double>(0,0), rs, Mat<double>(0,0,0));
	//allocating some temporary memory
	std::vector<double> tmpNB(uniqueCS.length()*nmodels);
	
	fitNBs_core(postMat, mus, rs, preproc, asMat(tmpNB, nmodels), nthreads);
	fitMultinoms_core(countsMat, postMat, ps, nthreads);
	return writeModels(mus, rs, ps);
}

// [[Rcpp::export]]
void fitNBs(
						Rcpp::NumericMatrix post, 
						Rcpp::NumericVector mus, 
						Rcpp::NumericVector rs,
						Rcpp::List ucs,
						int nthreads){
	
	Rcpp::IntegerVector uniqueCS = ucs["values"];
	Rcpp::IntegerVector map = ucs["map"];
	NMPreproc preproc(asVec(uniqueCS), asVec(map), Vec<double>(0,0));
	std::vector<double> tmpNB(uniqueCS.length()*mus.length());
	fitNBs_core(asMat(post), asVec(mus), asVec(rs), preproc, asMat(tmpNB, mus.length()), nthreads);
}

// [[Rcpp::export]]
Rcpp::NumericVector rowSumsDouble(Rcpp::NumericMatrix mat, int nthreads=1){
	std::vector<long double> acc(mat.nrow(), 0);
	rowSums<double, long double>(asMat(mat), asVec(acc), nthreads);
	Rcpp::NumericVector ret(mat.nrow());
	for (int i = 0, e = mat.nrow(); i < e; ++i){ret[i] = acc[i];}
	return ret;
}
