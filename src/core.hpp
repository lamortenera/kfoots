#include <omp.h>
#include <algorithm> 
#include <unordered_map>
#include <Rcpp.h>


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
};




template<typename TNumMat, typename TNumVec>
static void colSums(Mat<TNumMat> mat, Vec<TNumVec> vec, int nthreads){
	if (mat.ncol != vec.len) throw std::invalid_argument("provided vector has invalid length");

	TNumVec*  cs = vec.ptr;
	TNumMat* b = mat.ptr;
	int nrow = mat.nrow;
	int ncol = mat.ncol;
	
	#pragma omp parallel for default(none) firstprivate(cs, b, nrow, ncol) num_threads(std::max(1, nthreads))
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
	
	return (double)llik;
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

static inline double llik2posteriors_core(Mat<double> lliks, Vec<double> lmixcoeff, Mat<double> posteriors, int nthreads){
	long double tot = 0;
	int ncol = lliks.ncol;
	int nrow = lliks.nrow;
	
	#pragma omp parallel for reduction(+:tot) num_threads(std::max(1, nthreads))
	for (int col = 0; col < ncol; ++col){
		double* lliksCol = lliks.colptr(col);
		double* postCol = posteriors.colptr(col);
		//adding the mixing coefficients
		for (int row = 0; row < nrow; ++row){
			lliksCol[row] += lmixcoeff[row];
		}
		//getting maximum
		double cmax = lliksCol[0];
		for (int row = 1; row < nrow; ++row){
			if (lliksCol[row] > cmax) {
				cmax = lliksCol[row];
			} 
		}
		tot += cmax;
		double psum = 0;
		//subtracting maximum and exponentiating sumultaneously
		for (int row = 0; row < nrow; ++row){
			double tmp = exp(lliksCol[row] - cmax);
			postCol[row] = tmp;
			psum += tmp;
		}
		tot += log(psum);
		double* postCol_it = postCol;
		for (int row = 0; row < nrow; ++row, ++postCol_it){
			*postCol_it /= psum;
		}
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
