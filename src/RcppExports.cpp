// This file was generated by Rcpp::compileAttributes
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// forward_backward
List forward_backward(NumericVector initP, NumericMatrix trans, NumericMatrix lliks, IntegerVector seqlens, SEXP posteriors = R_NilValue, int nthreads = 1);
RcppExport SEXP kfoots_forward_backward(SEXP initPSEXP, SEXP transSEXP, SEXP lliksSEXP, SEXP seqlensSEXP, SEXP posteriorsSEXP, SEXP nthreadsSEXP) {
BEGIN_RCPP
    SEXP __sexp_result;
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< NumericVector >::type initP(initPSEXP );
        Rcpp::traits::input_parameter< NumericMatrix >::type trans(transSEXP );
        Rcpp::traits::input_parameter< NumericMatrix >::type lliks(lliksSEXP );
        Rcpp::traits::input_parameter< IntegerVector >::type seqlens(seqlensSEXP );
        Rcpp::traits::input_parameter< SEXP >::type posteriors(posteriorsSEXP );
        Rcpp::traits::input_parameter< int >::type nthreads(nthreadsSEXP );
        List __result = forward_backward(initP, trans, lliks, seqlens, posteriors, nthreads);
        PROTECT(__sexp_result = Rcpp::wrap(__result));
    }
    UNPROTECT(1);
    return __sexp_result;
END_RCPP
}
// viterbi
List viterbi(NumericVector initP, NumericMatrix trans, NumericMatrix lliks, NumericVector seqlens);
RcppExport SEXP kfoots_viterbi(SEXP initPSEXP, SEXP transSEXP, SEXP lliksSEXP, SEXP seqlensSEXP) {
BEGIN_RCPP
    SEXP __sexp_result;
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< NumericVector >::type initP(initPSEXP );
        Rcpp::traits::input_parameter< NumericMatrix >::type trans(transSEXP );
        Rcpp::traits::input_parameter< NumericMatrix >::type lliks(lliksSEXP );
        Rcpp::traits::input_parameter< NumericVector >::type seqlens(seqlensSEXP );
        List __result = viterbi(initP, trans, lliks, seqlens);
        PROTECT(__sexp_result = Rcpp::wrap(__result));
    }
    UNPROTECT(1);
    return __sexp_result;
END_RCPP
}
// orderColumns
Rcpp::IntegerVector orderColumns(Rcpp::IntegerMatrix mat);
RcppExport SEXP kfoots_orderColumns(SEXP matSEXP) {
BEGIN_RCPP
    SEXP __sexp_result;
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< Rcpp::IntegerMatrix >::type mat(matSEXP );
        Rcpp::IntegerVector __result = orderColumns(mat);
        PROTECT(__sexp_result = Rcpp::wrap(__result));
    }
    UNPROTECT(1);
    return __sexp_result;
END_RCPP
}
// llik2posteriors
Rcpp::List llik2posteriors(Rcpp::NumericMatrix lliks, Rcpp::NumericVector mix_coeff, SEXP posteriors = R_NilValue, int nthreads = 1);
RcppExport SEXP kfoots_llik2posteriors(SEXP lliksSEXP, SEXP mix_coeffSEXP, SEXP posteriorsSEXP, SEXP nthreadsSEXP) {
BEGIN_RCPP
    SEXP __sexp_result;
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type lliks(lliksSEXP );
        Rcpp::traits::input_parameter< Rcpp::NumericVector >::type mix_coeff(mix_coeffSEXP );
        Rcpp::traits::input_parameter< SEXP >::type posteriors(posteriorsSEXP );
        Rcpp::traits::input_parameter< int >::type nthreads(nthreadsSEXP );
        Rcpp::List __result = llik2posteriors(lliks, mix_coeff, posteriors, nthreads);
        PROTECT(__sexp_result = Rcpp::wrap(__result));
    }
    UNPROTECT(1);
    return __sexp_result;
END_RCPP
}
// mapToUnique
Rcpp::List mapToUnique(Rcpp::IntegerVector values);
RcppExport SEXP kfoots_mapToUnique(SEXP valuesSEXP) {
BEGIN_RCPP
    SEXP __sexp_result;
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type values(valuesSEXP );
        Rcpp::List __result = mapToUnique(values);
        PROTECT(__sexp_result = Rcpp::wrap(__result));
    }
    UNPROTECT(1);
    return __sexp_result;
END_RCPP
}
// getMultinomConst
Rcpp::NumericVector getMultinomConst(Rcpp::IntegerMatrix counts, int nthreads = 1);
RcppExport SEXP kfoots_getMultinomConst(SEXP countsSEXP, SEXP nthreadsSEXP) {
BEGIN_RCPP
    SEXP __sexp_result;
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< Rcpp::IntegerMatrix >::type counts(countsSEXP );
        Rcpp::traits::input_parameter< int >::type nthreads(nthreadsSEXP );
        Rcpp::NumericVector __result = getMultinomConst(counts, nthreads);
        PROTECT(__sexp_result = Rcpp::wrap(__result));
    }
    UNPROTECT(1);
    return __sexp_result;
END_RCPP
}
// sumAt
Rcpp::NumericVector sumAt(Rcpp::NumericVector values, Rcpp::IntegerVector map, int size, bool zeroIdx = false);
RcppExport SEXP kfoots_sumAt(SEXP valuesSEXP, SEXP mapSEXP, SEXP sizeSEXP, SEXP zeroIdxSEXP) {
BEGIN_RCPP
    SEXP __sexp_result;
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< Rcpp::NumericVector >::type values(valuesSEXP );
        Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type map(mapSEXP );
        Rcpp::traits::input_parameter< int >::type size(sizeSEXP );
        Rcpp::traits::input_parameter< bool >::type zeroIdx(zeroIdxSEXP );
        Rcpp::NumericVector __result = sumAt(values, map, size, zeroIdx);
        PROTECT(__sexp_result = Rcpp::wrap(__result));
    }
    UNPROTECT(1);
    return __sexp_result;
END_RCPP
}
// colSumsInt
Rcpp::IntegerVector colSumsInt(Rcpp::IntegerMatrix nums, int nthreads = 1);
RcppExport SEXP kfoots_colSumsInt(SEXP numsSEXP, SEXP nthreadsSEXP) {
BEGIN_RCPP
    SEXP __sexp_result;
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< Rcpp::IntegerMatrix >::type nums(numsSEXP );
        Rcpp::traits::input_parameter< int >::type nthreads(nthreadsSEXP );
        Rcpp::IntegerVector __result = colSumsInt(nums, nthreads);
        PROTECT(__sexp_result = Rcpp::wrap(__result));
    }
    UNPROTECT(1);
    return __sexp_result;
END_RCPP
}
// colSumsDouble
Rcpp::NumericVector colSumsDouble(Rcpp::NumericMatrix nums, int nthreads = 1);
RcppExport SEXP kfoots_colSumsDouble(SEXP numsSEXP, SEXP nthreadsSEXP) {
BEGIN_RCPP
    SEXP __sexp_result;
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type nums(numsSEXP );
        Rcpp::traits::input_parameter< int >::type nthreads(nthreadsSEXP );
        Rcpp::NumericVector __result = colSumsDouble(nums, nthreads);
        PROTECT(__sexp_result = Rcpp::wrap(__result));
    }
    UNPROTECT(1);
    return __sexp_result;
END_RCPP
}
// nbinomLoglik
Rcpp::NumericVector nbinomLoglik(Rcpp::IntegerVector counts, double mu, double r, int nthreads = 1);
RcppExport SEXP kfoots_nbinomLoglik(SEXP countsSEXP, SEXP muSEXP, SEXP rSEXP, SEXP nthreadsSEXP) {
BEGIN_RCPP
    SEXP __sexp_result;
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type counts(countsSEXP );
        Rcpp::traits::input_parameter< double >::type mu(muSEXP );
        Rcpp::traits::input_parameter< double >::type r(rSEXP );
        Rcpp::traits::input_parameter< int >::type nthreads(nthreadsSEXP );
        Rcpp::NumericVector __result = nbinomLoglik(counts, mu, r, nthreads);
        PROTECT(__sexp_result = Rcpp::wrap(__result));
    }
    UNPROTECT(1);
    return __sexp_result;
END_RCPP
}
// optimFun
double optimFun(Rcpp::IntegerVector counts, double mu, double r, Rcpp::NumericVector posteriors, int nthreads = 1);
RcppExport SEXP kfoots_optimFun(SEXP countsSEXP, SEXP muSEXP, SEXP rSEXP, SEXP posteriorsSEXP, SEXP nthreadsSEXP) {
BEGIN_RCPP
    SEXP __sexp_result;
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type counts(countsSEXP );
        Rcpp::traits::input_parameter< double >::type mu(muSEXP );
        Rcpp::traits::input_parameter< double >::type r(rSEXP );
        Rcpp::traits::input_parameter< Rcpp::NumericVector >::type posteriors(posteriorsSEXP );
        Rcpp::traits::input_parameter< int >::type nthreads(nthreadsSEXP );
        double __result = optimFun(counts, mu, r, posteriors, nthreads);
        PROTECT(__sexp_result = Rcpp::wrap(__result));
    }
    UNPROTECT(1);
    return __sexp_result;
END_RCPP
}
// fitMultinom
Rcpp::NumericVector fitMultinom(Rcpp::IntegerMatrix counts, Rcpp::NumericVector posteriors, int nthreads = 1);
RcppExport SEXP kfoots_fitMultinom(SEXP countsSEXP, SEXP posteriorsSEXP, SEXP nthreadsSEXP) {
BEGIN_RCPP
    SEXP __sexp_result;
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< Rcpp::IntegerMatrix >::type counts(countsSEXP );
        Rcpp::traits::input_parameter< Rcpp::NumericVector >::type posteriors(posteriorsSEXP );
        Rcpp::traits::input_parameter< int >::type nthreads(nthreadsSEXP );
        Rcpp::NumericVector __result = fitMultinom(counts, posteriors, nthreads);
        PROTECT(__sexp_result = Rcpp::wrap(__result));
    }
    UNPROTECT(1);
    return __sexp_result;
END_RCPP
}
// lLik
Rcpp::NumericVector lLik(Rcpp::IntegerMatrix counts, Rcpp::List model, SEXP ucs = R_NilValue, SEXP mConst = R_NilValue, int nthreads = 1);
RcppExport SEXP kfoots_lLik(SEXP countsSEXP, SEXP modelSEXP, SEXP ucsSEXP, SEXP mConstSEXP, SEXP nthreadsSEXP) {
BEGIN_RCPP
    SEXP __sexp_result;
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< Rcpp::IntegerMatrix >::type counts(countsSEXP );
        Rcpp::traits::input_parameter< Rcpp::List >::type model(modelSEXP );
        Rcpp::traits::input_parameter< SEXP >::type ucs(ucsSEXP );
        Rcpp::traits::input_parameter< SEXP >::type mConst(mConstSEXP );
        Rcpp::traits::input_parameter< int >::type nthreads(nthreadsSEXP );
        Rcpp::NumericVector __result = lLik(counts, model, ucs, mConst, nthreads);
        PROTECT(__sexp_result = Rcpp::wrap(__result));
    }
    UNPROTECT(1);
    return __sexp_result;
END_RCPP
}
// lLikMat
Rcpp::NumericMatrix lLikMat(Rcpp::IntegerMatrix counts, Rcpp::List models, SEXP ucs = R_NilValue, SEXP mConst = R_NilValue, SEXP lliks = R_NilValue, int nthreads = 1);
RcppExport SEXP kfoots_lLikMat(SEXP countsSEXP, SEXP modelsSEXP, SEXP ucsSEXP, SEXP mConstSEXP, SEXP lliksSEXP, SEXP nthreadsSEXP) {
BEGIN_RCPP
    SEXP __sexp_result;
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< Rcpp::IntegerMatrix >::type counts(countsSEXP );
        Rcpp::traits::input_parameter< Rcpp::List >::type models(modelsSEXP );
        Rcpp::traits::input_parameter< SEXP >::type ucs(ucsSEXP );
        Rcpp::traits::input_parameter< SEXP >::type mConst(mConstSEXP );
        Rcpp::traits::input_parameter< SEXP >::type lliks(lliksSEXP );
        Rcpp::traits::input_parameter< int >::type nthreads(nthreadsSEXP );
        Rcpp::NumericMatrix __result = lLikMat(counts, models, ucs, mConst, lliks, nthreads);
        PROTECT(__sexp_result = Rcpp::wrap(__result));
    }
    UNPROTECT(1);
    return __sexp_result;
END_RCPP
}
// pwhichmax
Rcpp::IntegerVector pwhichmax(Rcpp::NumericMatrix posteriors, int nthreads = 1);
RcppExport SEXP kfoots_pwhichmax(SEXP posteriorsSEXP, SEXP nthreadsSEXP) {
BEGIN_RCPP
    SEXP __sexp_result;
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type posteriors(posteriorsSEXP );
        Rcpp::traits::input_parameter< int >::type nthreads(nthreadsSEXP );
        Rcpp::IntegerVector __result = pwhichmax(posteriors, nthreads);
        PROTECT(__sexp_result = Rcpp::wrap(__result));
    }
    UNPROTECT(1);
    return __sexp_result;
END_RCPP
}
// fitNB_inner
Rcpp::List fitNB_inner(Rcpp::IntegerVector counts, Rcpp::NumericVector posteriors, double initR = -1);
RcppExport SEXP kfoots_fitNB_inner(SEXP countsSEXP, SEXP posteriorsSEXP, SEXP initRSEXP) {
BEGIN_RCPP
    SEXP __sexp_result;
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type counts(countsSEXP );
        Rcpp::traits::input_parameter< Rcpp::NumericVector >::type posteriors(posteriorsSEXP );
        Rcpp::traits::input_parameter< double >::type initR(initRSEXP );
        Rcpp::List __result = fitNB_inner(counts, posteriors, initR);
        PROTECT(__sexp_result = Rcpp::wrap(__result));
    }
    UNPROTECT(1);
    return __sexp_result;
END_RCPP
}
// fitModels
Rcpp::List fitModels(Rcpp::IntegerMatrix counts, Rcpp::NumericMatrix posteriors, Rcpp::List models, SEXP ucs = R_NilValue, int nthreads = 1);
RcppExport SEXP kfoots_fitModels(SEXP countsSEXP, SEXP posteriorsSEXP, SEXP modelsSEXP, SEXP ucsSEXP, SEXP nthreadsSEXP) {
BEGIN_RCPP
    SEXP __sexp_result;
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< Rcpp::IntegerMatrix >::type counts(countsSEXP );
        Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type posteriors(posteriorsSEXP );
        Rcpp::traits::input_parameter< Rcpp::List >::type models(modelsSEXP );
        Rcpp::traits::input_parameter< SEXP >::type ucs(ucsSEXP );
        Rcpp::traits::input_parameter< int >::type nthreads(nthreadsSEXP );
        Rcpp::List __result = fitModels(counts, posteriors, models, ucs, nthreads);
        PROTECT(__sexp_result = Rcpp::wrap(__result));
    }
    UNPROTECT(1);
    return __sexp_result;
END_RCPP
}
// fitNBs
void fitNBs(Rcpp::NumericMatrix post, Rcpp::NumericVector mus, Rcpp::NumericVector rs, Rcpp::List ucs, int nthreads);
RcppExport SEXP kfoots_fitNBs(SEXP postSEXP, SEXP musSEXP, SEXP rsSEXP, SEXP ucsSEXP, SEXP nthreadsSEXP) {
BEGIN_RCPP
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type post(postSEXP );
        Rcpp::traits::input_parameter< Rcpp::NumericVector >::type mus(musSEXP );
        Rcpp::traits::input_parameter< Rcpp::NumericVector >::type rs(rsSEXP );
        Rcpp::traits::input_parameter< Rcpp::List >::type ucs(ucsSEXP );
        Rcpp::traits::input_parameter< int >::type nthreads(nthreadsSEXP );
        fitNBs(post, mus, rs, ucs, nthreads);
    }
    return R_NilValue;
END_RCPP
}
// rowSumsDouble
Rcpp::NumericVector rowSumsDouble(Rcpp::NumericMatrix mat, int nthreads = 1);
RcppExport SEXP kfoots_rowSumsDouble(SEXP matSEXP, SEXP nthreadsSEXP) {
BEGIN_RCPP
    SEXP __sexp_result;
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type mat(matSEXP );
        Rcpp::traits::input_parameter< int >::type nthreads(nthreadsSEXP );
        Rcpp::NumericVector __result = rowSumsDouble(mat, nthreads);
        PROTECT(__sexp_result = Rcpp::wrap(__result));
    }
    UNPROTECT(1);
    return __sexp_result;
END_RCPP
}
