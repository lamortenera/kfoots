#include <Rcpp.h>
#include "core.cpp"
#include <algorithm> 


static inline double llik2posteriors_core(Mat<double> lliks, Vec<double> mix_coeff, Mat<double> posteriors, int nthreads){
    long double tot = 0;
    int ncol = lliks.ncol;
    int nrow = lliks.nrow;
    
    std::vector<long double> mix_coeff_acc(nrow, 0);
    
    //transform the mix_coeff taking the log
    for (int row = 0; row < nrow; ++row){
        mix_coeff[row] = log(mix_coeff[row]);
    }
    
    #pragma omp parallel num_threads(std::max(1, nthreads))
    {
        long double thread_tot = 0;
        std::vector<long double> thread_mix_coeff_acc(nrow, 0); 
        #pragma omp for nowait
        for (int col = 0; col < ncol; ++col){
            double* lliksCol = lliks.colptr(col);
            double* postCol = posteriors.colptr(col);
            double cmax = -std::numeric_limits<double>::infinity();
            //adding the mixing coefficients and getting maximum
            for (int row = 0; row < nrow; ++row){
                postCol[row] = lliksCol[row] + mix_coeff[row];
                if (postCol[row] > cmax) cmax = postCol[row];
            }
            thread_tot += cmax;
            double psum = 0;
            //subtracting maximum and exponentiating sumultaneously
            for (int row = 0; row < nrow; ++row){
                postCol[row] = exp(postCol[row] - cmax);
                psum += postCol[row];
            }
            thread_tot += log(psum);
            for (int row = 0; row < nrow; ++row){
                postCol[row] /= psum;
                thread_mix_coeff_acc[row] += postCol[row];
            }
        }
        //protected access to the shared variables
        #pragma omp critical
        {
            tot += thread_tot;
            for (int row = 0; row < nrow; ++row){
                mix_coeff_acc[row] += thread_mix_coeff_acc[row];
            }
        }
    }
    
    //normalizing mix coeff
    long double norm = 0;
    for (int row = 0; row < nrow; ++row){
        norm += mix_coeff_acc[row];
    }
    for (int row = 0; row < nrow; ++row){
        mix_coeff[row] = (double)(mix_coeff_acc[row]/norm);
    }
    
    
    return (double) tot;
}

// [[Rcpp::export]]
Rcpp::List llik2posteriors(Rcpp::NumericMatrix lliks, Rcpp::NumericVector mix_coeff, Rcpp::NumericMatrix posteriors, int nthreads=1){
    if (lliks.nrow() != posteriors.nrow() || lliks.ncol() != posteriors.ncol()) Rcpp::stop("lliks and posteriors matrix don't have the same format!");
    if (mix_coeff.length() != lliks.nrow()) Rcpp::stop("mix_coeff doens't match with the provided matrices");
    
    //copy the vector
    Rcpp::NumericVector new_mix_coeff(mix_coeff.begin(), mix_coeff.end());
    
    double tot = llik2posteriors_core(asMat(lliks), asVec(new_mix_coeff), asMat(posteriors), nthreads);
    
    return Rcpp::List::create(
                                    Rcpp::Named("posteriors")=posteriors,
                                    Rcpp::Named("tot_llik")=tot,
                                    Rcpp::Named("new_mix_coeff")=new_mix_coeff);
}


//' Group unique values of a vector
//'
//' @param v a vector of integers. If they are not integers they will be
//'     casted to integers.
//' @return a list with the following items:
//'        \item{values}{unique and sorted values of \code{v}}
//'        \item{map}{a vector such that \code{v[i] = values[map[i]+1]} for every i}
//'    @export
// [[Rcpp::export]]
Rcpp::List mapToUnique(Rcpp::IntegerVector values){
    Rcpp::IntegerVector map(values.length());
    
    Vec<int> valuesVec = asVec(values);
    Vec<int> mapVec = asVec(map);
    std::vector<int> uniqueCS;
    map2unique_core(valuesVec, mapVec, uniqueCS);
    
    return Rcpp::List::create(Rcpp::Named("values")=Rcpp::IntegerVector(uniqueCS.begin(),uniqueCS.end()), Rcpp::Named("map")=map);
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
Rcpp::NumericVector getMultinomConstSW(SEXP counts, int nthreads=1){    return getMultinomConst_helper(asSWMat<int>(counts), nthreads);}


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

static inline void parseModel(Rcpp::List model, double* mu, double* r, double** ps, int* footlen){
    Rcpp::NumericVector pstmp = model["ps"];
    
    *mu = model["mu"];
    *r = model["r"];
    *ps = pstmp.begin();
    *footlen = pstmp.length();
}

//that's bad, you should change this and use parseModel
static inline void parseModels(Rcpp::List models, Vec<double> mus, Vec<double> rs, Mat<double> ps){
    unsigned int footsize = sizeof(double)*ps.nrow;
    for (int i = 0; i < models.length(); ++i){
        Rcpp::List model = models[i];
        if (mus.ptr != 0){
            mus[i] = model["mu"]; 
        }
        if (rs.ptr != 0){
            rs[i] = model["r"];
        }
        if (ps.ptr != 0){
            Rcpp::NumericVector currps = model["ps"];
            memcpy(ps.colptr(i), currps.begin(), footsize);
        }
    }
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
void lLikMat(    Rcpp::IntegerMatrix counts, Rcpp::List models, 
                    Rcpp::List ucs, Rcpp::NumericVector mConst, Rcpp::NumericVector lliks,
                    int nthreads=1){
    lLikMat_helper(asMat(counts), models, ucs, mConst, lliks, nthreads);
}

// [[Rcpp::export]]
void lLikGapMat(    SEXP counts, Rcpp::List models, Rcpp::List ucs,
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
Rcpp::List fitNB_inner(Rcpp::IntegerVector counts, Rcpp::NumericVector posteriors, double initR=-1, double tol=1e-8, int nthreads=1){
    if (counts.length() != posteriors.length()) Rcpp::stop("counts and posteriors don't match");
    double mu = -1;
    double r = -1;
    fitNB_core(asVec(counts), asVec(posteriors), &mu, &r, initR, tol, nthreads);
    
    return Rcpp::List::create(Rcpp::Named("mu")=mu, Rcpp::Named("r")=r);
}


template<template <typename> class TMat>
inline Rcpp::List fitModels_helper(TMat<int> counts, Rcpp::NumericVector posteriors, Rcpp::List models, Rcpp::List ucs, std::string type="indep", double tol=1e-8, int nthreads=1){
    int nmodels = models.length();
    int footlen = counts.nrow;
    
    if (    counts.ncol*nmodels != posteriors.length()){
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
    
     
    if (type=="nofit" || type=="pois"){
        //fit only the mus
        //if pois is selected, set r to Inf
        //inverse transformation: the column sums from the unique column sums
        GapVec<int> colsums(uniqueCS.begin(), map.begin(), map.length());
        //fit only the mus
        fitMeans_core(colsums, postMat, mus, nthreads);
        if (type=="pois"){ for (int i = 0; i < nmodels; ++i) {rs[i] = INFINITY; } }
    }  else if (type=="indep" || type=="dep"){
        //allocating some temporary memory
        std::vector<double> cpost_mem(uniqueCS.length()*nmodels);
        Mat<double> cpost = asMat(cpost_mem, uniqueCS.length());
        //collapse columns of the posterior matrix with the same total count
        //std::cout << "collapsing posteriors" << std::endl;
        collapsePosteriors_core(cpost, postMat, preproc, nthreads);
        if (type == "dep"){//constrain the NBs to have the same r
            double r = rs[0];
            //std::cout << "calling fitNBs_1r_core" << std::endl;
            fitNBs_1r_core(preproc.uniqueCS, cpost, mus, &r, tol, nthreads);
            for (int i = 0; i < nmodels; ++i){ rs[i] = r; }
        } else {//each NB has its own r
            //transpose the cpost matrix
            //std::cout << "transposing matrix" << std::endl;
            std::vector<double> tcpost_mem(cpost_mem.size());
            Mat<double> tcpost = asMat(tcpost_mem, nmodels); int nu = cpost.ncol;
            for (int m = 0; m < nmodels; ++m){
                double* S = cpost.ptr + m;
                double* D = tcpost.colptr(m);
                for (int i = 0; i < nu; ++i, S += nmodels) D[i] = *S;
            }
            //std::cout << "calling fitNBs_core" << std::endl;
            fitNBs_core(preproc.uniqueCS, tcpost, mus, rs, tol, nthreads);
        }
    } else Rcpp::stop("Invalid fitting method provided: must be one among 'indep', 'nofit', 'pois' and 'dep'.");
    
    fitMultinoms_core(counts, postMat, ps, nthreads);
    return writeModels(mus, rs, ps);
}



// [[Rcpp::export]]
Rcpp::List fitModels(Rcpp::IntegerMatrix counts, Rcpp::NumericVector posteriors, Rcpp::List models, Rcpp::List ucs, std::string type="indep", double tol=1e-8, int nthreads=1){
    return fitModels_helper(asMat(counts), posteriors, models, ucs, type, tol, nthreads);
}

// [[Rcpp::export]]
Rcpp::List fitModelsGapMat(SEXP counts, Rcpp::NumericVector posteriors, Rcpp::List models, Rcpp::List ucs, std::string type="indep", double tol=1e-8, int nthreads=1){
    return fitModels_helper(asGapMat<int>(counts), posteriors, models, ucs, type, tol, nthreads);
}

// [[Rcpp::export]]
void checkInterrupt(){
    R_CheckUserInterrupt();
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
