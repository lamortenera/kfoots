#include <time.h>

// [[Rcpp::export]]
double getRealSeconds(){
	timespec t;
	clock_gettime(CLOCK_REALTIME, &t);
	return t.tv_sec + t.tv_nsec*1e-9;
}
