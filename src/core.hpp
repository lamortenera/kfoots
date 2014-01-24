#include <omp.h>
#include <algorithm> 
#include <unordered_map>
#include <Rcpp.h>
#include <math.h>
#include "optim.cpp"

//Matrix and vector wrappers for pointers
template<typename TType>
struct Vec {
	TType* ptr;
	int len;
	
	Vec(){}
	
	Vec(TType* _ptr, int _len):
		ptr(_ptr), len(_len){}
	
	inline TType operator[] (int i) const {return ptr[i];}
	inline TType& operator[] (int i){return ptr[i];}
	
	Vec subset(int start, int end){
		return Vec(ptr + start, end - start);
	}
};

template<typename TType>
struct Mat {
	TType* ptr;
	int nrow;
	int ncol;
	
	Mat(){}
	
	Mat(TType* _ptr, int _nrow, int _ncol):
		ptr(_ptr), nrow(_nrow), ncol(_ncol){}
	
	inline TType operator[] (int i) const {return ptr[i];}
	inline TType& operator[] (int i){return ptr[i];}
	
	inline TType operator() (int row, int col) const {return ptr[row + col*nrow];}
	inline TType& operator() (int row, int col) {return ptr[row + col*nrow];}
	
	Mat subsetCol(int colStart, int colEnd){
		return Mat(ptr + nrow*colStart, nrow, colEnd-colStart);
	}
	
	
	inline TType get(int row, int col){
		return ptr[row + col*nrow];
	}
	
	inline TType* colptr(int col){
		return ptr + col*nrow;
	}
	
	inline Vec<TType> getCol(int col){
		return Vec<TType>(ptr + col*nrow, nrow);
	}
};




template<typename TNumMat, typename TNumVec>
static void colSums(Mat<TNumMat> mat, Vec<TNumVec> vec, int nthreads){
	if (mat.ncol != vec.len) throw std::invalid_argument("provided vector has invalid length");

	TNumVec*  cs = vec.ptr;
	TNumMat* b = mat.ptr;
	int nrow = mat.nrow;
	int ncol = mat.ncol;
	
	#pragma omp parallel for schedule(static) num_threads(std::max(1, nthreads))
	for (int col = 0; col < ncol; ++col){
		TNumMat* ptr = b + col*nrow;
		TNumMat tmp = 0;
		for (int row = 0; row < nrow; ++row){
			tmp += *ptr++;
		}
		cs[col] = tmp;
	}
}


//this is not going to work with vector<bool>,
//because it doesn't contain a pointer to bools
template<typename TType>
inline Vec<TType> asVec(std::vector<TType>& v){
	return Vec<TType>(v.data(), v.size());
}

template<typename TType, int RType>
inline Vec<TType> asVec(Rcpp::Vector<RType>& v){
	return Vec<TType>(v.begin(), v.length());
}

template<typename TType, int RType>
inline Mat<TType> asMat(Rcpp::Matrix<RType>& m){
	return Mat<TType>(m.begin(), m.nrow(), m.ncol());
}

template<typename TType>
inline Mat<TType> asMat(std::vector<TType>& v, int ncol){
	if (v.size() % ncol != 0) throw std::invalid_argument("number of columns must be a divisor of vector length");
	return Mat<TType>(v.data(), v.size()/ncol, ncol);
}



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

struct Avatar {
	int count;
	int pos;
	Avatar(int _count, int _pos): count(_count), pos(_pos){}
};

static inline bool avatarSorter(const Avatar& a, const Avatar& b){
	return b.count > a.count;
}

//uniqueCS should be an empty vector. It also works if map and values 
//wrap the same pointer (so if values is a temporary vector that later becomes map).
static void map2unique_core(Vec<int> values, Vec<int> map, std::vector<int>& uniqueCS){
	//this section is not parallelized:
	//it should be done just once and the running time
	//should be very low, independent on the number of rows.
	
	//sort the column counts keeping track of the original position
	Avatar empty(0,0);
	std::vector<Avatar> avatars(values.len, empty);
	for (int i = 0, e = values.len; i < e; ++i){
		avatars[i].count = values[i];
		avatars[i].pos = i;
		
	}
	std::sort(avatars.begin(), avatars.end(), avatarSorter);
	
	//fill in uniqueCS and map
	int lastVal = avatars[0].count;
	uniqueCS.push_back(lastVal);
	map[avatars[0].pos] = 0;
	
	for (int i = 0, e = avatars.size(); i < e; ++i){
		Avatar& a = avatars[i];
		if (a.count != lastVal) {
			lastVal = a.count;
			uniqueCS.push_back(lastVal);
		}
		map[a.pos] = uniqueCS.size()-1;
	}
}

static void getMultinomConst_core(Mat<int> counts, Vec<double> multinomConst, int nthreads){
	//compute the log multinomial for each column
	CachedLFact lfact(0.75);
	int* i_counts = counts.ptr;
	double* i_mc = multinomConst.ptr;
	int ncol = counts.ncol;
	int nrow = counts.nrow;
	
	#pragma omp parallel for default(none) firstprivate(lfact, i_counts, i_mc, nrow, ncol) num_threads(std::max(1, nthreads))
	for (int col = 0; col < ncol; ++col){
		int* i_ccounts = i_counts + col*nrow;
		double tmp = 0;
		int colsum = 0;
		for (int row = 0; row < nrow; ++row){
			colsum += *i_ccounts;
			tmp -= lfact(*i_ccounts++);
		}
		i_mc[col] = tmp + lfact(colsum);
	}
}

static inline void nbinomLoglik_core(Vec<int> counts, double mu, double r, Vec<double> lliks, int nthreads){
	int e = counts.len;
	double* i_lliks = lliks.ptr;
	int* i_counts = counts.ptr;
	#pragma omp parallel for default(none) firstprivate(e, i_lliks, i_counts, mu, r) num_threads(nthreads)
	for (int i = 0; i < e; ++i){
		i_lliks[i] = Rf_dnbinom_mu(i_counts[i], r, mu, 1);
	}
}


static inline double optimFun_core(Vec<int> counts, double mu, double r, Vec<double> posteriors, int nthreads){
	
	int e = counts.len;
	double* i_post = posteriors.ptr;
	int* i_counts = counts.ptr;
	long double llik = 0;
	#pragma omp parallel for default(none) firstprivate(e, i_post, i_counts, mu, r) reduction(+:llik) num_threads(nthreads)
	for (int i = 0; i < e; ++i){
		double tmp = Rf_dnbinom_mu(i_counts[i], r, mu, 1)*(i_post[i]);
		if (!std::isnan(tmp)){ llik += tmp; }
	}
	//std::cout << "OptimFun_core @: " << r << "->" << -llik <<std::endl;
	//std::cout << -llik << std::endl;
	//std::cout << "optimFun_core @: " << r << "->" << -llik <<std::endl;
	return (double)llik;
}


//data for optimization function.
//It is always assumed that mu is also the mean count.
struct optimData {
	Vec<int> counts;
	Vec<double> posteriors;
	double mu;
	double spost;
	int nthreads;
};

//optimization function: it's very fast but it does not work
/* It's a problem of numerical stability...
 negative log likelihood function without the term due to the factorials
 * (as they don't play a role in the optimization)
 
static double fn(int n, double* logr, void* data){
	optimData* info = (optimData*) data;
	double mu = info->mu;
	double spost = info->spost;
	double ret;
	double r = exp(*logr);
	if (!std::isfinite(r)){//r==+Inf, poisson case
		ret = spost*mu*(1 - log(mu));
		
	} else if (r == 0){//degenerate distribution concentrated at 0
		if (mu == 0) ret = 0; 
		else ret = INFINITY;
	} else {//normal case
		ret = spost*(-Rf_lgammafn(r) + mu*log(mu/(mu+r)) + r*log(r/(mu+r)));
		
		long double part2 = 0;
		int e = info->counts.len;
		double* post = info->posteriors.ptr;
		int* counts = info->counts.ptr;
		
		
		for (int i = 0; i < e; ++i){
			double tmp = Rf_lgammafn(counts[i]+r)*post[i];
			if (!std::isnan(tmp)){ part2 += tmp; }
		}
		
		ret += part2;
	}
	
	std::cout << "fn @: " << r << "->" << -ret <<std::endl;
	
		
	return -ret;
}
*/

//optimization function (for 2 different APIs)

//brent API
static double fn1d(double logr, void* data){
	optimData* info = (optimData*) data;
	double mu = info->mu;
	double r = exp(logr);
	double* post = info->posteriors.ptr;
	int* counts = info->counts.ptr;
	int e = info->counts.len;
	int nthreads = info->nthreads;
	
	long double llik = 0;
	if (r == 0){//degenerate distribution concentrated at 0
		if (mu == 0) llik = 0; 
		else llik = -INFINITY;
	} else if (!std::isfinite(r)){//r==+Inf, poisson case

		#pragma omp parallel for reduction(+:llik) num_threads(nthreads)
		for (int i = 0; i < e; ++i){
			double tmp = Rf_dpois(counts[i], mu, 1)*post[i];
			if (!std::isnan(tmp)){ llik += tmp; }
		}
	} else {//normal case
		
		#pragma omp parallel for reduction(+:llik) num_threads(nthreads)
		for (int i = 0; i < e; ++i){
			double tmp = Rf_dnbinom_mu(counts[i], r, mu, 1)*post[i];
			if (!std::isnan(tmp)){ llik += tmp; }
		}
	}
	//std::cout << "fn @: " << r << "->" << -llik <<std::endl;
	
	return -llik;
}

//lbfgsf API
static double fn(int n, double* logr, void* data){
	return fn1d(*logr, data);
}


//derivative of the optimization function
static void dfn(int n, double* x, double* ret, void* data){
	/*
	optimData* info = (optimData*) data;
	double mu = info->mu;
	double r = exp(*x);
	double spost = info->spost;
	
	double part1 = spost*(-Rf_digamma(r) + log(r/(mu+r)));
	
	long double part2 = 0;
	int e = info->counts.len;
	double* post = info->posteriors.ptr;
	int* counts = info->counts.ptr;
	
	
	for (int i = 0; i < e; ++i){
		double tmp = Rf_digamma(counts[i]+r)*post[i];
		if (!std::isnan(tmp)){ part2 += tmp; }
	}
	
	*ret = -(part1 + part2)*r;
	*/
	//something is wrong with the real derivative... no idea what
	//0.001 is the same parameter used by R
	
	double x1 = *x + 0.001;
	double x2 = *x - 0.001;
	double nd = (fn(n, &x1, data) - fn(n, &x2, data))/0.002;
	//double r = exp(*x);
	
	//std::cout << "dfn: " << r << "->" << nd << std::endl;
	
	*ret = nd;
	
	
	
	if(!std::isfinite(*ret))
		throw std::invalid_argument("non-finite value in derivative of optimization function");
}

static inline void fitNB_core(Vec<int> counts, Vec<double> posteriors, double* mu, double* r, double initR, int nthreads=1){
	//get weighted average
	long double sum_cp = 0;
	long double sum_p = 0;
	int* c = counts.ptr;
	double* p = posteriors.ptr;
	for (int i = 0, e = counts.len; i < e; ++i, ++c, ++p){
		sum_cp += (*c) * (*p);
		sum_p += *p;
	}
	*mu = sum_cp/sum_p;
	
	if (initR <= 0){
		//find a suitable initial value for r:
		//estimate and match empirical second moment
		long double sum_c2p = 0;
		c = counts.ptr;
		p = posteriors.ptr;
		for (int i = 0, e = counts.len; i < e; ++i, ++c, ++p){
			sum_c2p += (*c) * (*c) * (*p);
		}
		initR = (*mu) * (*mu) / (sum_c2p/sum_p  -  *mu * (*mu + 1));
	}
	
	if (initR < 0 || !std::isfinite(initR)){
		//either the second moment matched above does not have overdispersion
		// or the initial value was +Inf
		initR = DBL_MAX; //poisson case
	}
	
	optimData info;
	info.counts = counts;
	info.posteriors = posteriors;
	info.mu = *mu;
	info.spost = sum_p;
	info.nthreads = nthreads;
	//default used by R
	//optimization is done in the log space
	initR = log(initR);
	
	//double maxfn = 0;
	
	//lbfgsb_wrapper(&fn, &dfn, &info, &initR, &maxfn, 0, 0);
	//bfgs_wrapper(&fn, &dfn, &info, &initR, &maxfn);
	//log(r1), log(r1) + 0.1, fn, &info, tol
	initR = brent_wrapper(initR, initR + 0.1, fn1d, &info, 1e-8);
	
	*r = exp(initR);
}



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

struct NegMultinom {
	//associated preprocessing data
	typedef NMPreproc Preproc;
	
	//model parameters
	int len;
	double r;
	double mu;
	double* p;
	
	NegMultinom(double _r, double _mu, Vec<double> _p){
		r = _r;
		mu = _mu;
		p = _p.ptr;
		len = _p.len;
	}
	
	
	void getLlik(Mat<int> counts, Vec<double> llik, Preproc& preproc, int nthreads){
		if (len != counts.nrow) throw std::invalid_argument("the counts matrix doesn't have the right number of rows");
		if (counts.ncol != preproc.map.len) throw std::invalid_argument("the preprocessed data were not computed on the same count matrix");
		
		nthreads = std::max(1, nthreads);
		//compute all the log neg binom on the unique column sums
		std::vector<double> logNBinom(preproc.uniqueCS.len);
		nbinomLoglik_core(preproc.uniqueCS, mu, r, asVec<double>(logNBinom), nthreads);
		
		//pre-compute all the log(p) 
		std::vector<double> logP(len);
		for (int i = 0; i < len; ++i){
			logP[i] = log(p[i]);
		}
		
		//iterate through each column
		double* i_nb = logNBinom.data();
		int* i_counts = counts.ptr;
		double* i_mtnm = preproc.multinomConst.ptr;
		int* i_map = preproc.map.ptr;
		double* i_logp = logP.data();
		double* i_llik = llik.ptr;
		int ncol = counts.ncol;
		int nrow = counts.nrow;
				
		#pragma omp parallel for default(none) firstprivate(i_nb, i_counts, i_mtnm, i_map, i_logp, i_llik, ncol, nrow) num_threads(nthreads)
		for (int col = 0; col < ncol; ++col){
			double tmp = i_nb[i_map[col]] + i_mtnm[col];
			int* i_ccounts = i_counts + col*nrow;
			double* i_llogp = i_logp;
			for (int row = 0; row < nrow; ++row, ++i_ccounts, ++i_llogp){
				double ttemp = (*i_ccounts)*(*i_llogp);
				if (!std::isnan(ttemp)){
					tmp += ttemp;
				}
			}
			i_llik[col] = tmp;
		}
	}
};


static void lLikMat_core(Mat<int> counts, Vec<double> mus, Vec<double> rs, Mat<double> ps, Mat<double> llik, NMPreproc& preproc, Mat<double> tmpNB, int nthreads){
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
	
	
	#pragma omp parallel num_threads(nthreads)
	{
		//compute all the log neg binom on the unique column sums
		//Here the outer loop is on the models because I think it divides the
		//computation more evenly (even if we need to collapse): uniqueCS
		//is sorted...
		#pragma omp for schedule(static) collapse(2) nowait
		for (int p = 0; p < nmodels; ++p){
			for (int c = 0; c < nUCS; ++c){
				tmpNB(p,c) = Rf_dnbinom_mu(uniqueCS[c], rs[p], mus[p], 1);
			}
		}
		
		//pre-compute all the log(p)
		#pragma omp for schedule(static) 
		for (int i = 0; i < logPsSize; ++i){
			logPs[i] = log(ps[i]);
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
					double ttemp = countsCol[row]*logp[row];
					if (!std::isnan(ttemp)){
						tmp += ttemp;
					}
				}
				llikCol[mod] = tmp;
			}
		}
	}
}


static inline void fitMultinom_core(Mat<int> counts, Vec<double> posteriors, Vec<double> fit, int nthreads){
	int* i_counts = counts.ptr;
	double* i_post = posteriors.ptr;
	double* i_fit = fit.ptr;
	int nrow = counts.nrow;
	int ncol = counts.ncol;
		
	if (nthreads > 1){
		#pragma omp parallel firstprivate(i_counts, i_post, i_fit, nrow, ncol) num_threads(nthreads)
		{
			std::vector<double> acc(nrow, 0);
			double* i_locFit = acc.data();
			#pragma omp for nowait
			for (int col = 0; col < ncol; ++col){
				double post = i_post[col];
				if (post != 0){
					int* i_colCounts = i_counts + col*nrow;
					for (int row = 0; row < nrow; ++row){
						i_locFit[row] += i_colCounts[row]*post;
					}
				}
			}
			#pragma omp critical
			{
				for (int row = 0; row < nrow; ++row){
					i_fit[row] += i_locFit[row];
				}
			}
		}
	} else {
		
		for (int col = 0; col < ncol; ++col){
			double post = i_post[col];
			if (post != 0){
				for (int row = 0; row < nrow; ++row, ++i_counts){
					i_fit[row] += *i_counts*post;
				}
			} else { i_counts += nrow; }
		}
	}
	
	double sum = 0;
	for (int row = 0; row < nrow; ++row){sum += fit[row];}
	for (int row = 0; row < nrow; ++row){fit[row] /= sum;}
}


//here vec must be clean at the beginning
template<typename TNumMat, typename TNumVec>
static void rowSums(Mat<TNumMat> mat, Vec<TNumVec> vec, int nthreads){
	if (mat.nrow != vec.len) throw std::invalid_argument("provided vector has invalid length");

	int nrow = mat.nrow;
	int ncol = mat.ncol;
	
	#pragma omp parallel num_threads(std::max(1, nthreads))
	{
		std::vector<TNumVec> acc(nrow, 0);
		TNumVec* accBegin = acc.data();
		#pragma omp for schedule(static) nowait
		for (int col = 0; col < ncol; ++col){
			TNumMat* matCol = mat.colptr(col);
			TNumVec* accIter = accBegin;
			for (int row = 0; row < nrow; ++row){
				*accIter++ += *matCol++;
			}
		}
		#pragma omp critical
		{
			for (int row = 0; row < nrow; ++row){
				vec[row] += acc[row];
			}
		}
	}
}


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

static inline double  forward_backward_core(Vec<double> initP, Mat<double> trans, Mat<double> lliks, Vec<int> seqlens, Mat<double> posteriors, Mat<double> new_trans, int nthreads){
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


struct colPtr {
	int* ptr;
	int colref;
	colPtr(int* _ptr, int _colref): ptr(_ptr), colref(_colref){}
};

struct colSorter{
	int nrow;
	
	inline bool operator() (const colPtr& cp1, const colPtr& cp2) {
		int* ptr1 = cp1.ptr;
		int* ptr2 = cp2.ptr;
		
		for (int i = 0; i < nrow; ++i){
			int diff = ptr2[i] - ptr1[i];
			if (diff!=0){
				return diff > 0;
			}
		}
		
		return false;
	}
	
	colSorter(int _nrow) : nrow(_nrow){}
};

static inline void orderColumns_core(Mat<int> mat, Vec<int> vec){
	int nc = mat.ncol;
	colSorter srtr(mat.nrow);
	colPtr foo(0,0);
	std::vector<colPtr> cols(nc, foo);
	for (int i = 0; i < nc; ++i){cols[i] = colPtr(mat.colptr(i), i);}
	std::sort(cols.begin(), cols.end(), srtr);
	for (int i = 0; i < nc; ++i){vec[i] = cols[i].colref+1;}
}

static inline void pwhichmax_core(Mat<double> posteriors, Vec<int> clusters, int nthreads){
	int ncol = posteriors.ncol;
	int nrow = posteriors.nrow;
	#pragma omp parallel for num_threads(nthreads)
	for (int col = 0; col < ncol; ++col){
		double* postCol = posteriors.colptr(col);
		double max = *postCol;
		int whichmax = 1;
		for (int row = 1; row < nrow; ++row){
			++postCol;
			if (*postCol > max){
				max = *postCol;
				whichmax = row + 1;
			}
		}
		clusters[col] = whichmax;
	}
}


static void fitNBs_core(Mat<double> posteriors, Vec<double> mus, Vec<double> rs, NMPreproc& preproc, Mat<double> tmpNB, int nthreads){
	//std::cout << "fitNBs_core" << std::endl;
	int ncol = posteriors.ncol;
	int nmod = posteriors.nrow;
	if (mus.len != nmod || rs.len != nmod || tmpNB.ncol*tmpNB.nrow != nmod*preproc.uniqueCS.len){
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
	//std::cout << "Threads per model: " << threads_per_model << std::endl;
	omp_set_num_threads(nthreads);
	omp_set_nested(true);
	//make sure we're not using more than nthreads threads
	omp_set_dynamic(true);
	//std::cout << "starting main loop" << std::endl;
	//fitting the negative binomials
	
	#pragma omp parallel for schedule(dynamic) num_threads(nthreads_outer)
	for (int mod = 0; mod < nmod; ++mod){
		//std::cout << "mod: " << mod << std::endl;
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
		//std::cout << "calling fitNB_core" << std::endl;
		fitNB_core(values, Vec<double>(tmpNBCol, tmpNB.nrow), &mus[mod], &rs[mod], rs[mod], nthreads_inner);
	}
}

static void fitMultinoms_core(Mat<int> counts, Mat<double> posteriors, Mat<double> ps, int nthreads){
	//std::cout << "fitMultinom_core" << std::endl;
	int nmod = posteriors.nrow;
	int nrow = counts.nrow;
	int ncol = counts.ncol;
	
	#pragma omp parallel num_threads(nthreads)
	{
		//fitting the multinomial 
		std::vector<double> tmp_ps_std(nrow*nmod, 0);
		Mat<double> tmp_ps = asMat(tmp_ps_std, nmod);
		#pragma omp for schedule(static) nowait
		for (int col = 0; col < ncol; ++col){
			//This you could probably do better with BLAS...
			double* postCol = posteriors.colptr(col);
			int* countCol = counts.colptr(col);
			for (int mod = 0; mod < nmod; ++mod, ++postCol){
				double post = *postCol;
				double* psCol = tmp_ps.colptr(mod);
				for (int row = 0; row < nrow; ++row){
					psCol[row] += countCol[row]*post;
				}
			}
		}
		
		
		#pragma omp critical
		{
			for (int i = 0, e = nrow*nmod; i < e; ++i){
				ps[i] += tmp_ps[i];
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
