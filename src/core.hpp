#include <omp.h>
#include <algorithm> 
#include <Rcpp.h>
#include <boost/unordered_map.hpp>


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
	
	Mat subsetCol(int colStart, int colEnd){
		return Mat(ptr + nrow*colStart, nrow, colEnd-colStart);
	}
	
	
	inline TType get(int row, int col){
		return ptr[row + col*nrow];
	}
	
	
};


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



struct CachedLFact{
	boost::unordered_map<int, double> cache;
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

static inline double llik2posteriors_core(Mat<double> lliks, Mat<double> posteriors, int nthreads){
	long double tot = 0;
	double* i_lliks = lliks.ptr;
	double* i_post = posteriors.ptr;
	int ncol = lliks.ncol;
	int nrow = lliks.nrow;
	
	#pragma omp parallel for default(none) firstprivate(i_lliks, i_post, ncol, nrow) reduction(+:tot) num_threads(std::max(1, nthreads))
	for (int col = 0; col < ncol; ++col){
		double* i_llliks = i_lliks + col*nrow;
		double* i_ppost = i_post + col*nrow;
		double cmax = i_llliks[0];
		for (int row = 1; row < nrow; ++row){ if (i_llliks[row] > cmax) { cmax = i_llliks[row]; } }
		tot += cmax;
		double psum = 0;
		for (int row = 0; row < nrow; ++row){
			double tmp = exp(i_llliks[row] - cmax);
			i_ppost[row] = tmp;
			psum += tmp;
		}
		tot += log(psum);
		for (int row = 0; row < nrow; ++row, ++i_ppost){
			*i_ppost = *i_ppost/psum;
		}
	}
	
	return (double) tot;
}

static inline void fitMultinom_core(Mat<int> counts, Vec<double> posteriors, Vec<double>(fit), int nthreads){
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
