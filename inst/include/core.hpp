#include "array.cpp"
#include <omp.h>
#include <algorithm> 
#include <unordered_map>
#include <Rcpp.h>
#include <math.h>
#include "optim.cpp"



/* MAP 2 UNIQUE ROUTINES */
struct Avatar {
	int count;
	int pos;
	Avatar(int _count, int _pos): count(_count), pos(_pos){}
};

static inline bool avatarSorter(const Avatar& a, const Avatar& b){
	return b.count > a.count;
}

//values is the input variable, map and uvalues are the output variables
//map and values can also wrap the same pointer
static void map2unique_core(Vec<int> values, Vec<int> map, std::vector<int>& uvalues){
	//this function is not parallelized:
	//it should be done just once and the running time should be very low
	
	if (values.len <= 0){
		return;//otherwise the code breaks with vectors of length 0
	}
	
	
	//sort the counts keeping track of the original position
	Avatar empty(0,0);
	std::vector<Avatar> avatars(values.len, empty);
	for (int i = 0, e = values.len; i < e; ++i){
		avatars[i].count = values[i];
		avatars[i].pos = i;
		
	}
	std::sort(avatars.begin(), avatars.end(), avatarSorter);
	
	//fill in uniquevalues and map
	int lastVal = avatars[0].count;
	uvalues.push_back(lastVal);
	map[avatars[0].pos] = 0;
	
	for (int i = 1, e = avatars.size(); i < e; ++i){
		Avatar& a = avatars[i];
		if (a.count != lastVal) {
			lastVal = a.count;
			uvalues.push_back(lastVal);
		}
		map[a.pos] = uvalues.size()-1;
	}
}

//subset is a index vector passed from R that we don't want to copy, so subtract 1 to all indices
static void subsetM2U_core(Vec<int> uvalues, Vec<int> map, Vec<int> subset, std::vector<int>& newuvalues, Vec<int> newmap){
	if (uvalues.len > 2*subset.len){
		/* the subset is very small, it is not worth going through the "old" unique values */
		//call the standard map2unique_core, but prepare the newmap vector
		for (int i = 0, e = subset.len; i < e; ++i){
			newmap[i] = uvalues[map[subset[i]-1]];
		}
		map2unique_core(newmap, newmap, newuvalues);
	} else {
		/* use the "old" unique values as a hashmap to avoid sorting */
		std::vector<int> hashmap(uvalues.len, -1);
		//mark with a 1 the counts that are present
		for (int i = 0, e = subset.len; i < e; ++i){
			hashmap[map[subset[i]-1]] = 1;
		}
		//fill the new uvalues vector and update the indices in hashmap
		for (int i = 0, e = uvalues.len; i < e; ++i){
			if (hashmap[i] > 0){
				newuvalues.push_back(uvalues[i]);
				hashmap[i] = newuvalues.size() - 1;
			}
		}
		//set up the newmap vector
		for (int i = 0, e = subset.len; i < e; ++i){
			newmap[i] = hashmap[map[subset[i]-1]];
		}
	}
}


/* MULTINOM CONST ROUTINES */

struct CachedLFact{
	std::unordered_map<int, double> cache;
	CachedLFact(double load_factor){
		cache.max_load_factor(load_factor);
	}
	
	inline double operator()(int n){
		if (n <= 1){ return 0; }
		double cachedValue = cache[n];
		if (cachedValue == 0){//value not present in cache
			cache[n] = cachedValue = Rf_lgammafn(n + 1);
		}
		return cachedValue;
	}
};

//it can be implemented specifically for SWMat. The calls to 
//the lgamma functions would be the same, but less calls to lfact
//and faster colSums
template<template <typename> class TMat>
static void getMultinomConst_core(TMat<int> counts, Vec<double> multinomConst, int nthreads){
	//compute the log multinomial for each column
	CachedLFact lfact(0.75);
	int ncol = counts.ncol;
	int nrow = counts.nrow;
	
	#pragma omp parallel for firstprivate(lfact) num_threads(std::max(1, nthreads))
	for (int col = 0; col < ncol; ++col){
		int* countsCol = counts.colptr(col);
		double tmp = 0;
		int colsum = 0;
		for (int row = 0; row < nrow; ++row){
			int c = countsCol[row];
			colsum += c;
			tmp -= lfact(c);
		}
		multinomConst[col] = tmp + lfact(colsum);
	}
}


/* FITTING ROUTINES */

//this will slow down a bit, but it's safe... hopefully they will fix it in R 3.1...
//the caller has the responsibility that mu and r are >= 0
//this thing can be broken...
#define lognbinom(c, mu, r) (std::isfinite(mu*r)?Rf_dnbinom_mu(c, r, mu, 1):Rf_dpois(c, mu, 1))

struct NMPreproc {
	Vec<int> uniqueCS;
	Vec<int> map;
	Vec<double> multinomConst;
	NMPreproc(){}
	NMPreproc(Vec<int> _uniqueCS, Vec<int> _map, Vec<double> _multinomConst){
		uniqueCS = _uniqueCS;
		map = _map;
		multinomConst = _multinomConst;
	}
};

//data for optimization function.
//It is always assumed that mu is also the mean count.
struct optimData {
	Vec<int> counts;
	Vec<double> posteriors;
	double mu;
	double spost;
	int nthreads;
};

//optim fun for Brent
static double fn1d(double logr, void* data){
	optimData* info = (optimData*) data;
	double mu = info->mu;
	double r = exp(logr);
	double* post = info->posteriors.ptr;
	int* counts = info->counts.ptr;
	int e = info->counts.len;
	int nthreads = info->nthreads;
	
	long double llik = 0;
	#pragma omp parallel for schedule(static) reduction(+:llik) num_threads(nthreads)
	for (int i = 0; i < e; ++i){
		if (post[i] > 0){
			llik += lognbinom(counts[i], mu, r)*post[i];
		}
	}
	
	return -llik;
}

//you need the guarantee that the r that you get will be better or equal to initR
//if initR is acceptable (not negative and not infinity) it must be one of the two
//points used for brent
static inline void fitNB_core(Vec<int> counts, Vec<double> posteriors, double* mu, double* r, double initR, int nthreads=1){
	//tolerance
	double tol=1e-8;
	//get weighted average
	long double sum_c2p = 0;
	long double sum_cp = 0;
	long double sum_p = 0;
	int* c = counts.ptr;
	double* p = posteriors.ptr;
	for (int i = 0, e = counts.len; i < e; ++i, ++c, ++p){
		double P = *p, C = *c;
		sum_p += P; P *= C;
		sum_cp += P;
		sum_c2p += P*C;
	}
	*mu = sum_cp/sum_p;
	
	//the r that matches the sample variance
	double guessR = (*mu) * (*mu) / (sum_c2p/sum_p  -  *mu * (*mu + 1));
	//low sample variance -> poisson case
	if (guessR < 0 || !std::isfinite(guessR)){ 
		guessR = DBL_MAX;
	}
	if (initR < 0){
		initR = guessR;
	} else if (!std::isfinite(initR)){
		//this is just to avoid to work with infinities, 
		//but it should give the same score as infinity
		initR = DBL_MAX; 
	}
	
	//optimization is done in the log space to avoid boundary constraints
	guessR = log(guessR);
	initR = log(initR);
	//initial points too close to each other
	if (fabs(guessR - initR) < tol*fabs(guessR + initR)/2){
		guessR = initR*(1 - tol);
	}
	
	
	optimData info;
	info.counts = counts;
	info.posteriors = posteriors;
	info.mu = *mu;
	info.spost = sum_p;
	info.nthreads = nthreads;
	
	initR = brent_wrapper(initR, guessR, fn1d, &info, tol);
		
	*r = exp(initR);
}


static void fitNBs_core(Mat<double> posteriors, Vec<double> mus, Vec<double> rs, NMPreproc& preproc, Mat<double> tmpNB, int nthreads){
	int ncol = posteriors.ncol;
	int nmod = posteriors.nrow;
	if (mus.len != nmod || rs.len != nmod || tmpNB.ncol*tmpNB.nrow != nmod*preproc.uniqueCS.len || ncol != preproc.map.len){
		throw std::invalid_argument("invalid parameters passed to fitNBs_core");
	}
	Vec<int> map = preproc.map;
	Vec<int> values = preproc.uniqueCS;
	
	//make sure that tmpNB here has the desired format 
	//(transpose of the one used in llikMat)
	tmpNB.ncol = nmod;
	tmpNB.nrow = values.len;
	
	//for the fitNB function call, I didn't find anything better than
	//nested parallel regions... this happens when nmod < nthreads
	int nthreads_outer = nmod > nthreads? nthreads : nmod;
	int nthreads_inner = ceil(((double)nthreads)/nmod);
	omp_set_nested(true);
	//make sure we're not using more than nthreads threads
	omp_set_dynamic(true);
	//fitting the negative binomials
	
	
	#pragma omp parallel for schedule(dynamic) num_threads(nthreads_outer)
	for (int mod = 0; mod < nmod; ++mod){
		//set up the tmpNB matrix: it will contain the
		//posteriors wrt the unique counts
		double* tmpNBCol = tmpNB.colptr(mod);
		memset(tmpNBCol, 0, tmpNB.nrow*sizeof(double));
		//this should be parallelized on the models, because otherwise tmpNBCol is
		//shared
		for (int col = 0; col < ncol; ++col){
			tmpNBCol[map[col]] += posteriors(mod, col);
		}
		//call the optimization procedure. This will use up to threads_per_model threads
		fitNB_core(values, Vec<double>(tmpNBCol, tmpNB.nrow), &mus[mod], &rs[mod], rs[mod], nthreads_inner);
	}
}

//ps matrix needs to be clean at the beginning
template<template <typename> class TMat>
static void fitMultinoms_core(TMat<int> counts, Mat<double> posteriors, Mat<double> ps, int nthreads){
	int nmod = posteriors.nrow;
	int nrow = counts.nrow;
	int ncol = counts.ncol;
	if (ncol <= 0 || nrow <= 0 || nmod <= 0 || posteriors.ncol != ncol || ps.nrow != nrow || ps.ncol != nmod){
		throw std::invalid_argument("invalid parameters passed to fitMultinoms_core");
	}
	
	#pragma omp parallel num_threads(nthreads)
	{
		std::vector<double> loc_ps_std(nrow*nmod, 0);
		Mat<double> loc_ps = asMat(loc_ps_std, nmod);
		if (nmod == 1){//separate case with one model for efficiency
			#pragma omp for schedule(static) nowait
			for (int col = 0; col < ncol; ++col){
				double post = posteriors[col];
				if (post > 0){
					int* countsCol = counts.colptr(col);
					for (int row = 0; row < nrow; ++row){
						loc_ps[row] += countsCol[row]*post;
					}
				}
			}
		} else {
			#pragma omp for schedule(static) nowait
			for (int col = 0; col < ncol; ++col){
				//This you could probably do better with BLAS...
				double* postCol = posteriors.colptr(col);
				int* countsCol = counts.colptr(col);
				for (int mod = 0; mod < nmod; ++mod, ++postCol){
					double post = *postCol;
					if (post > 0){
						double* loc_psCol = loc_ps.colptr(mod);
						for (int row = 0; row < nrow; ++row){
							loc_psCol[row] += countsCol[row]*post;
						}
					}
				}
			}
		}
		
		#pragma omp critical
		{
			for (int i = 0, e = nrow*nmod; i < e; ++i){
				ps[i] += loc_ps[i];
			}
		}
		#pragma omp barrier
		//ps contains now the non-normalized optimal ps parameter, normalize!
		//parallelization is probably overkill, but can it be worse than without?
		
		#pragma omp for schedule(static)
		for (int mod = 0; mod < nmod; ++mod){
			double* psCol = ps.colptr(mod);
			double sum = 0;
			for (int row = 0; row < nrow; ++row){sum += psCol[row];}
			for (int row = 0; row < nrow; ++row){psCol[row] /= sum;}
		}
	}
}



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




/* EXPERIMENTAL LOG LIKELIHOOD ROUTINES  */

static inline double dotprod(int* v1, double* v2, int len){
	int startup = len % 5;
	double sum = 0; int i = 0;
	for (; i < startup; ++i){ sum += v1[i]*v2[i]; }
	for (; i < len; i += 5){
		sum += v1[i]*v2[i] + v1[i+1]*v2[i+1] + v1[i+2]*v2[i+2] + v1[i+3]*v2[i+3] + v1[i+4]*v2[i+4]; 
	}
	return sum;
}
static void lLik_nbinom(double mu, double r, Vec<int> uniqueCS, Vec<double> tmpNB, int nthreads){
	if (tmpNB.len != uniqueCS.len) throw std::invalid_argument("the preprocessed data were not computed on the same count matrix");
	/* MAIN LOOP */
	int nUCS = uniqueCS.len;
	#pragma omp parallel for schedule(static) num_threads(nthreads)
	for (int c = 0; c < nUCS; ++c){
		tmpNB[c] = lognbinom(uniqueCS[c], mu, r);
	}
}

template<template <typename> class TMat>
static void lLik_multinom(TMat<int> counts, Vec<double> ps, Vec<double> llik, Vec<int> mapp, Vec<double> tmpNB, Vec<double> mconst, int nthreads){
	if (ps.len != counts.nrow) throw std::invalid_argument("incoherent models provided");
	if (counts.ncol != mapp.len) throw std::invalid_argument("the preprocessed data were not computed on the same count matrix");
	
	int ncol = counts.ncol;
	int nrow = counts.nrow;
	int* map = mapp.ptr;
	double* llikptr = llik.ptr;
	double* mtnmConst = mconst.ptr;
	std::vector<double> logPsSTD(nrow);
	Vec<double> logPs = asVec(logPsSTD);
	
	//pre-compute all the log(p)
	for (int i = 0; i < nrow; ++i){ logPs[i] = log(ps[i]);}
	
	/* MAIN LOOP */
	#pragma omp parallel for schedule(static) num_threads(nthreads)
	for (int col = 0; col < ncol; ++col){
		llikptr[col] = tmpNB[map[col]] + mtnmConst[col] + dotprod(counts.colptr(col), logPs.ptr, nrow);
	}
}




/* LOG LIKELIHOOD ROUTINES  */

template<template <typename> class TMat>
static void lLikMat_core(TMat<int> counts, Vec<double> mus, Vec<double> rs, Mat<double> ps, Mat<double> llik, NMPreproc& preproc, Mat<double> tmpNB, int nthreads){
	if (rs.len != mus.len || mus.len != ps.ncol || ps.nrow != counts.nrow) throw std::invalid_argument("incoherent models provided");
	if (counts.ncol != preproc.map.len) throw std::invalid_argument("the preprocessed data were not computed on the same count matrix");
	
	nthreads = std::max(1, nthreads);
	int nmodels = mus.len;
	int ncol = counts.ncol;
	int nrow = counts.nrow;
	int nUCS = preproc.uniqueCS.len;
	int logPsSize = nrow*nmodels;
	int* uniqueCS = preproc.uniqueCS.ptr;
	int* map = preproc.map.ptr;
	double* mtnmConst = preproc.multinomConst.ptr;
	std::vector<double> logPsSTD(nrow*nmodels);
	Mat<double> logPs = asMat(logPsSTD, nmodels);
	
	//pre-compute all the log(p)
	for (int i = 0; i < logPsSize; ++i){
		logPs[i] = log(ps[i]);
	}
	
	/* MAIN LOOP */
	if (nmodels==1){ //separate case where there is only one model, for efficiency
		double mu = mus[0], r = rs[0]; double* llikptr = llik.ptr; 
		#pragma omp parallel num_threads(nthreads)
		{
			#pragma omp for schedule(static)
			for (int c = 0; c < nUCS; ++c){
				tmpNB[c] = lognbinom(uniqueCS[c], mu, r);
			}
			
			//contribution of the multinomial
			#pragma omp for schedule(static)
			for (int col = 0; col < ncol; ++col){
				double tmp = tmpNB[map[col]] + mtnmConst[col];
				int* currcount = counts.colptr(col);
				const double* currlogp = logPs.ptr;
				for (int row = 0; row < nrow; ++row, ++currcount, ++currlogp){
					int c = *currcount;
					if (c != 0){
						tmp += c*(*currlogp);
					}
				}
				llikptr[col] = tmp;
			}
		}
		
	} else { 
		#pragma omp parallel num_threads(nthreads)
		{
			//compute all the log neg binom on the unique column sums
			//Here the outer loop is on the models because I think it divides the
			//computation more evenly (even if we need to collapse): uniqueCS
			//is sorted...
			#pragma omp for schedule(static) collapse(2)
			for (int p = 0; p < nmodels; ++p){
				for (int c = 0; c < nUCS; ++c){
					tmpNB(p,c) = lognbinom(uniqueCS[c], mus[p], rs[p]);
				}
			}
			
			//compute and add contribution of the multinomial 
			#pragma omp for schedule(static) 
			for (int col = 0; col < ncol; ++col){
				int* countsCol = counts.colptr(col);;
				double* llikCol = llik.colptr(col);
				double mtnm = mtnmConst[col];
				double* tmpNBcol = tmpNB.colptr(map[col]);
				
				for (int mod = 0; mod < nmodels; ++mod){
					double tmp = tmpNBcol[mod] + mtnm;
					double* logp = logPs.colptr(mod);
					for (int row = 0; row < nrow; ++row){
						int c = countsCol[row];
						if (c != 0){
							tmp += c*logp[row];
						}
					}
					llikCol[mod] = tmp;
				}
			}
		}
	}
}


/* LOG LIKELIHOOD TO POSTERIORS ROUTINES */

//at the beginning mixcoeff contains the mixture coefficient,
//at the end it contains the newly-trained coefficients
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
			//adding the mixing coefficients
			for (int row = 0; row < nrow; ++row){
				lliksCol[row] += mix_coeff[row];
			}
			//getting maximum
			double cmax = lliksCol[0];
			for (int row = 1; row < nrow; ++row){
				if (lliksCol[row] > cmax) {
					cmax = lliksCol[row];
				} 
			}
			thread_tot += cmax;
			double psum = 0;
			//subtracting maximum and exponentiating sumultaneously
			for (int row = 0; row < nrow; ++row){
				double tmp = exp(lliksCol[row] - cmax);
				postCol[row] = tmp;
				psum += tmp;
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

static inline double forward_backward_core(Vec<double> initP, Mat<double> trans, Mat<double> lliks, Vec<int> seqlens, Mat<double> posteriors, Mat<double> new_trans, int nthreads){
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
	
	
	#pragma omp parallel num_threads(nthreads)
	{
		//each thread gets one copy of these temporaries
		std::vector<double> tmpSTD(nrow*nrow, 0); 
		std::vector<double> thread_new_transSTD(nrow*nrow, 0); 
		std::vector<double> backward(nrow, 0);
		std::vector<double> new_backward(nrow, 0);
		long double thread_llik = 0;
		
		Mat<double> tmp = asMat(tmpSTD, nrow);
		Mat<double> thread_new_trans = asMat(thread_new_transSTD, nrow);
		
		/* transform the lliks matrix to the original space (exponentiate). 
		 * A column-specific factor is multiplied to obtain a better numerical
		 * stability */
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
		 * Chunks might have very different lengths (that's why dynamic schedule). */
		#pragma omp for schedule(dynamic,1) nowait
		for (int o = 0; o < nchunk; ++o){
			int chunk_start = chunk_starts[o];
			int chunk_end =  chunk_start + seqlens[o];
			
			/* FORWARD LOOP */
			/* first iteration is from fictitious start state */
			{
				double cf = 0;//scaling factor
				double* emissprob = lliks.colptr(chunk_start);
				double* forward = posteriors.colptr(chunk_start);
				for (int r = 0; r < nrow; ++r){
					double p = emissprob[r]*initP[r];
					forward[r] = p;
					cf += p;
				}
				for (int r = 0; r < nrow; ++r){
					forward[r] = forward[r]/cf;
				}
				thread_llik += log(cf);
			}
			/* all other iterations */
			for (int i = chunk_start + 1; i < chunk_end; ++i){
				double cf = 0;//scaling factor
				double* emissprob = lliks.colptr(i);
				double* forward = posteriors.colptr(i);
				double* last_forward = posteriors.colptr(i-1);
			
				for (int t = 0; t < nrow; ++t){
					double* transcol = trans.colptr(t);
					double acc = 0;
					for (int s = 0; s < nrow; ++s){
						acc += last_forward[s]*transcol[s];
					}
					acc *= emissprob[t];
					forward[t] = acc;
					cf += acc;
				}
				for (int t = 0; t < nrow; ++t){
					forward[t] = forward[t]/cf;
				}
				thread_llik += log(cf);
			}
			
			/* BACKWARD LOOP */
			/* first iteration set backward to 1/k, 
			 * last column of posteriors is already ok */
			for (int r = 0; r < nrow; ++r){
				backward[r] = 1.0/nrow;
			}
			for (int i = chunk_end-2; i >= chunk_start; --i){
				double* emissprob = lliks.colptr(i+1);
				double* posterior = posteriors.colptr(i);
				double cf = 0;
				double norm = 0;
				/* joint probabilities and backward vector */
				for (int s = 0; s < nrow; ++s){
					//the forward variable is going to be overwritten with the posteriors
					double pc = posterior[s];
					double acc = 0;
					
					for (int t = 0; t < nrow; ++t){
						double p = trans(s, t)*emissprob[t]*backward[t];
						tmp(s, t) = pc*p;
						acc += p;
					}
					
					new_backward[s] = acc;
					cf += acc;
				}
				/* update backward vector */
				for (int s = 0; s < nrow; ++s){
					backward[s] = new_backward[s]/cf;
					norm += backward[s]*posterior[s];
				}
				/* update transition probabilities */
				for (int t = 0, e = nrow*nrow; t < e; ++t){
					thread_new_trans[t] += tmp[t]/(norm*cf);
				}
				/* get posteriors */
				for (int s = 0; s < nrow; ++s){
					posterior[s] = posterior[s]*backward[s]/norm;
				}
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
	// should I put it inside the parallel region?
	// The code to generate the static schedule might be 
	// slower than this loop....
	for (int row = 0; row < nrow; ++row){
		double sum = 0;
		for (int col = 0; col < nrow; ++col){sum += new_trans(row, col);}
		for (int col = 0; col < nrow; ++col){new_trans(row, col)/=sum;}
	}
	
	return (double) tot_llik;
}
