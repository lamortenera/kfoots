#include <Rcpp.h>
#include "core.cpp"
#include "schedule.cpp"

template <typename T>
static inline bool allequal(const T& a, const T& b) {return a==b;}
template <typename T, typename... S>
static inline bool allequal(const T& a, const T& b, const T& c, const S&... d){ 
    return a==b && allequal(a, c, d...); 
}

//temporary storage needed by fb_iter
struct FBtmp {
    int nstates;
    std::vector<double> mem;
    Vec<double> backward;
    Vec<double> new_backward;
    Mat<double> tmp;
    FBtmp(int ns) : nstates(ns), mem(ns*(ns+2)) {
        double* start = mem.data();
        backward = Vec<double>(start, nstates);
        new_backward = Vec<double>(start + nstates, nstates);
        tmp = Mat<double>(start + 2*nstates, nstates, nstates);
    }
};

//forward backward iteration for one sequence of observations
//posteriors: at the beginning it doesn't matter, at the end, the posterior probabilities
//eprobs: at the beginning and at the end, the emission probabilities
//eprobs and prosteriors have the same dimensions. 
//The rows are the states, the columns are the observations
//initP: at the beginning and at the end the initial probabilities
//new_initP: at the beginning it doesn't matter, at the end the fitted initial probabilitites
//trans: at the beginning and end, the transition probabilitites
//new_trans: at the beginning, it doesn't matter, at the end the new transition
//probabilities ARE ADDED to the initial values, unnormalized, to allow accumulation
//llik: at the beginning, it doesn't matter, at the end the log likelihood of
//this sequence is ADDED
//storage: temporary storage needed by the function. You can provided this arg
//for efficiency, but you don't need to.
inline void fb_iter(Mat<double> eprobs, Mat<double> posteriors, 
                    Vec<double> initP, Vec<double> new_initP, 
                    Mat<double> trans, Mat<double> new_trans, long double& llik,
                    FBtmp& storage){
    int nobs = eprobs.ncol;
    int nstates = eprobs.nrow;
    if (nobs != posteriors.ncol || 
        !allequal(nstates, posteriors.nrow, initP.len, new_initP.len, 
                    trans.ncol, trans.nrow, new_trans.nrow, new_trans.ncol, 
                    storage.nstates)){
        Rcpp::stop("invalid dimensions of input arrays");
    }
    if (nobs == 0) return;
    /* FORWARD LOOP */
    /* first iteration is from fictitious start state */
    {
        double cf = 0;//scaling factor
        double* emissprob = eprobs.colptr(0);
        double* forward = posteriors.colptr(0);
        for (int r = 0; r < nstates; ++r){
            double p = emissprob[r]*initP[r];
            forward[r] = p;
            cf += p;
        }
        if (cf==0) Rcpp::stop("underflow error");
        for (int r = 0; r < nstates; ++r){
            forward[r] = forward[r]/cf;
        }
        llik += log(cf);
    }
    /* all other iterations */
    for (int i = 0 + 1; i < nobs; ++i){
        double cf = 0;//scaling factor
        double* emissprob = eprobs.colptr(i);
        double* forward = posteriors.colptr(i);
        double* last_forward = posteriors.colptr(i-1);
    
        for (int t = 0; t < nstates; ++t){
            double* transcol = trans.colptr(t);
            double acc = 0;
            for (int s = 0; s < nstates; ++s){
                acc += last_forward[s]*transcol[s];
            }
            acc *= emissprob[t];
            forward[t] = acc;
            cf += acc;
        }
        if (cf==0) Rcpp::stop("underflow error");
        for (int t = 0; t < nstates; ++t){
            forward[t] = forward[t]/cf;
        }
        llik += log(cf);
    }
    
    /* BACKWARD LOOP */
    /* we don't keep the backward matrix, only a 'backward' column */
    /* this gets replaced by 'new_backward' at each iteration */
    /* first iteration set backward to 1/k, 
     * last column of posteriors is already ok */
    
    Vec<double> backward = storage.backward;
    Vec<double> new_backward = storage.new_backward;
    Mat<double> tmp = storage.tmp;
    
    for (int r = 0; r < nstates; ++r){
        backward[r] = 1.0/nstates;
    }
    for (int i = nobs-2; i >= 0; --i){
        double* emissprob = eprobs.colptr(i+1);
        double* posterior = posteriors.colptr(i);
        double cf = 0;
        double norm = 0;
        /* joint probabilities and backward vector */
        for (int s = 0; s < nstates; ++s){
            //the forward variable is going to be overwritten with the posteriors
            double pc = posterior[s];
            double acc = 0;
            
            for (int t = 0; t < nstates; ++t){
                double p = trans(s, t)*emissprob[t]*backward[t];
                tmp(s, t) = pc*p;
                acc += p;
            }
            
            new_backward[s] = acc;
            cf += acc;
        }
        if (cf==0) Rcpp::stop("underflow error");
        /* update backward vector */
        for (int s = 0; s < nstates; ++s){
            backward[s] = new_backward[s]/cf;
            norm += backward[s]*posterior[s];
        }
        /* update transition probabilities */
        for (int t = 0, e = nstates*nstates; t < e; ++t){
            new_trans[t] += tmp[t]/(norm*cf);
        }
        /* get posteriors */
        for (int s = 0; s < nstates; ++s){
            posterior[s] = posterior[s]*backward[s]/norm;
        }
    }
    /* set new_initP */
    double* posterior = posteriors.colptr(0);
    for (int r = 0; r < nstates; ++r){
        new_initP[r] = posterior[r];
    }
}

inline void fb_iter(Mat<double> eprobs, Mat<double> posteriors, 
                    Vec<double> initP, Vec<double> new_initP, 
                    Mat<double> trans, Mat<double> new_trans, long double& llik){
    FBtmp storage(eprobs.nrow);
    fb_iter(eprobs, posteriors, initP, new_initP, trans, new_trans, llik, storage);
}


static inline double fb_core(Mat<double> initPs, Mat<double> trans, Mat<double> lliks, Vec<int> seqlens, 
                             Mat<double> posteriors, Mat<double> new_trans, Mat<double> new_initPs, int nthreads){
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
    
    //figure out how to assign the chromosomes to the threads
    //covert seqlens to double
    std::vector<double> jobSize(nchunk);
    for (int i = 0; i < nchunk; ++i) jobSize[i] = seqlens[i];
    //get the assignments
    std::vector<int> breaks = scheduleJobs(asVec(jobSize), nthreads);
    
    #pragma omp parallel num_threads(nthreads)
    {
        //each thread gets one copy of these temporaries
        std::vector<double> thread_new_transSTD(nrow*nrow, 0); 
        Mat<double> thread_new_trans = asMat(thread_new_transSTD, nrow);
        long double thread_llik = 0;
        FBtmp thread_tmp(nrow);
               
        /* transform the log likelihoods to probabilities (exponentiate). 
         * A column-specific factor is multiplied to obtain a better numerical
         * stability. This tends to be the bottle-neck of the whole
         * algorithm, but it is indispensable, and it scales well with the
         * number of cores.*/
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
         * Chunks might have very different lengths (that's why they have been scheduled). */
        #pragma omp for schedule(static) nowait
        for (int thread = 0; thread < nthreads; ++thread){
            for (int o = breaks[thread]; o < breaks[thread+1]; ++o){
                //o is the index that identifies the sequence
                int chunk_start = chunk_starts[o];
                int chunk_end =  chunk_start + seqlens[o];
                
                fb_iter(lliks.subsetCol(chunk_start, chunk_end), 
                        posteriors.subsetCol(chunk_start, chunk_end), 
                        initPs.getCol(o), new_initPs.getCol(o), trans, 
                        thread_new_trans, thread_llik, thread_tmp);
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
//' Warning: this function overwrites the lliks matrix. 
//' @param initP matrix of initial probabilities: each column corresponds to a sequence
//' @param trans transition matrix (rows are previous state, columns are next state)
//' @param lliks matrix with emission probabilities for each datapoint and each state.
//' Columns are datapoints and rows are states.
//' @param seqlens length of each subsequence of datapoints (set this to ncol(lliks)
//' if there is only one sequence).
//' @param posteriors the posteriors matrix where the posteriors will be written.
//' its value when the function is called does not matter, but it needs to have
//' the right dimensions (rows are states and columns are observations).
//' @param nthreads number of threads used. Sequences of observations are
//' processed independently by different threads (if \code{length(seqlens) > 1}).
//' @return a list with the following arguments:
//'    \item{posteriors}{posterior probability of being in a certain state for a certain datapoint.
//'     Same matrix used as input argument.}
//'    \item{tot_llik}{total log-likelihood of the data given the hmm model.}
//'    \item{new_trans}{update for the transition probabilities (it is already normalized).}
//' @export
// [[Rcpp::export]]
List forward_backward(NumericMatrix initP, NumericMatrix trans, NumericMatrix lliks, IntegerVector seqlens, NumericMatrix posteriors, int nthreads=1){
    int nmod = initP.nrow();
    double totlen = Rcpp::sum(seqlens);
    if (nmod != trans.nrow() || nmod != trans.ncol() || nmod != lliks.nrow() || nmod != posteriors.nrow()) Rcpp::stop("Unable to figure out the number of models");
    if (((double) lliks.ncol()) != totlen || ((double)posteriors.ncol()) != totlen) Rcpp::stop("Seqence lengths don't match with the provided matrices");
    if (initP.ncol() != seqlens.length()) Rcpp::stop("'initP' must have as many columns as the number of sequences");
    
    NumericMatrix newTrans(trans.nrow(), trans.ncol());
    NumericMatrix newInitP(initP.nrow(), initP.ncol());
    double tot_llik = fb_core(asMat(initP), asMat(trans), asMat(lliks), asVec(seqlens), asMat(posteriors), asMat(newTrans), asMat(newInitP), nthreads);
    return List::create(_("posteriors")=posteriors, _("tot_llik")=tot_llik, _("new_trans")=newTrans, _("new_initP")=newInitP);
}



//' Viterbi algorithm
//'
//' Standard viterbi algorithm in the log space
//' @param initP matrix of initial probabilities: each column corresponds to a sequence
//' @param trans transition matrix (rows are previous state, columns are next state)
//' @param lliks matrix with emission probabilities for each datapoint and each state.
//' Columns are datapoints and rows are states.
//' @param seqlens length of each subsequence of datapoints (set this to ncol(lliks)
//' if there is only one sequence).
//' @return a list with the following arguments:
//'    \item{vpath}{viterbi path}
//'    \item{vllik}{log-likelihood of the viterbi path}
//' @export
// [[Rcpp::export]]
List viterbi(NumericMatrix initP, NumericMatrix trans, NumericMatrix lliks, NumericVector seqlens){
    int nmod = initP.nrow();
    double totlen = Rcpp::sum(seqlens);
    if (nmod != trans.nrow() || nmod != trans.ncol() || nmod != lliks.nrow()) Rcpp::stop("Unable to figure out the number of models");
    if (((double) lliks.ncol()) != totlen) Rcpp::stop("Sequence lengths don't match with the provided matrix");
    
    int ncol = lliks.ncol();
    IntegerVector vpath(ncol);
    IntegerMatrix backtrack(nmod, max(seqlens));
    std::vector<long double> scores(nmod);
    std::vector<long double> new_scores(nmod);
    
    /* log-transform the transition probabilities */
    NumericMatrix ltrans(nmod,nmod);
    for (diter curr = ltrans.begin(), currt = trans.begin(); curr < ltrans.end(); ++curr, ++currt){
        *curr = log(*currt);
    }
    
    /* Viterbi independently on each chunk */
    double tot_maxscore = 0;
    for (int o = 0, chunk_start = 0; o < seqlens.length(); chunk_start += seqlens[o], ++o){
        int chunk_end = chunk_start + seqlens[o];
        /* dynamic programming */
        {
            MatrixColumn<REALSXP> llikcol = lliks.column(chunk_start);
            MatrixColumn<REALSXP> curr_initP = initP.column(o);
            for (int t = 0; t < nmod; ++t){
                scores[t] = llikcol[t] + log(curr_initP[t]);
            }
        }
        for (int i = chunk_start + 1; i < chunk_end; ++i){
            
            MatrixColumn<REALSXP> llikcol = lliks.column(i);
            MatrixColumn<INTSXP> backtrackcol = backtrack.column(i-chunk_start);
            
            for (int t = 0; t < nmod; ++t){
                int maxs = 0;
                long double maxscore = scores[0] + ltrans(0, t);
                for (int s = 1; s < nmod; ++s){
                    long double currscore = scores[s] + ltrans(s,t);
                    if (currscore > maxscore){
                        maxscore = currscore;
                        maxs = s;
                    }
                }
                backtrackcol[t] = maxs;
                new_scores[t] = llikcol[t] + maxscore;
            }
            
            memcpy(scores.data(), new_scores.data(), sizeof(long double)*nmod);
        }
        
        /* backtracking */
        int maxp = 0;
        double maxscore = scores[0];
        for (int p = 1; p < nmod; ++p){
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

/*
// [[Rcpp::export]]
Rcpp::IntegerVector orderColumns(Rcpp::IntegerMatrix mat){
    Rcpp::IntegerVector order(mat.ncol());
    orderColumns_core(asMat(mat), asVec(order));
    return order;
}
*/


// [[Rcpp::export]]
Rcpp::List testSchedule(Rcpp::NumericVector jobs, int nthreads, int type){
    std::vector<int> breaks(nthreads+1);
    Vec<double> jobSize = asVec(jobs);
    if (type == 0){
        scheduleNaive(jobSize, breaks);
    } else if (type == 1){
        scheduleGreedy(jobSize, breaks);
    } else if (type == 2){
        scheduleOptimal(jobSize, breaks);
    } else Rcpp::stop("invalid type");
    double makespan = getMakespan(jobSize, breaks);
    return Rcpp::List::create(  
        Rcpp::Named("makespan")=makespan, 
        Rcpp::Named("breaks")=Rcpp::wrap(breaks));
}

//static inline void collapsePosteriors_core(Mat<double> cpost, Mat<double> post, NMPreproc& preproc, int nthreads=1)

// [[Rcpp::export]]
Rcpp::NumericMatrix testColPost(Rcpp::NumericMatrix post, Rcpp::List m2u, int nthreads){
    Rcpp::IntegerVector values = Rcpp::as<Rcpp::IntegerVector>(m2u["values"]);
    Rcpp::IntegerVector map = Rcpp::as<Rcpp::IntegerVector>(m2u["map"]);
    if (post.ncol() != map.length()) Rcpp::stop("posteriors doesn't match with m2u");
    
    Rcpp::NumericMatrix smallerPost(post.nrow(), values.length());
    Vec<double> foo; NMPreproc preproc(asVec(values), asVec(map), foo);
    collapsePosteriors_core(asMat(smallerPost), asMat(post), preproc);
    return smallerPost;
}
