#include <float.h> 
#include <math.h> 
#include <string>
#include <stdexcept> 

extern "C" {
	#include <R.h>
	#include <R_ext/Applic.h>
}


/*
	* Just a wrapper around the optim function call in R
	* This is a lot less general:
	* 1. it uses only the "L-BFGS-B" optim method
	* 2. it uses only functions of one variable
	* 3. it does not use derivative information
	* In case of need, points 2 and 3 can be easily extended.
	* 
	* 
	* I abandoned the L-BFGS-B approach because very often it didn't converge.
	* Both from R and from C sometimes it starts cycling through the same values.
	* It could be that it is not well suited to one-dimensional problems,
	* (as the optim help message says) or that it is just buggy.
*/

/* Type of the optimization function:
	
	The first argument is the number of parameters in the second argument (always one here...).
	The second argument is the point where the function is evaluated.
	The third argument is a pointer passed down from the calling routine, normally used to carry auxiliary information. 
	The return value is the value of the function.
*/
typedef double optimfn(int, double *, void *);

/* Type of the gradient of the optimization function:
	
	The first argument is the number of parameters in the second argument.
	The second argument is the point where the function is evaluated.
	The third argument at the end contains the computed gradient vector.
	The fourth argument is a pointer passed down from the calling routine, normally used to carry auxiliary information. 
*/
typedef void optimgr(int, double *, double *, void *);


/*
*  x is the starting parameter on entry and x the final parameter on exit
*  fn is the optimization function
*  gr is the gradient function
*  lb is a pointer to the lower bound, or 0 if there are no lower bounds
*  ub is a pointer to the upper bound, or 0 if there are no upper bounds
*  fx at the end contains the attained minimum
*  ex carries external information for the function fn
*/


static inline void lbfgsb_wrapper(optimfn fn, optimgr gr, void* ex, double* x, double* fx, double* lb, double* ub){
	
	//dimension of the parameter space
	int n = 1; 
	/*
	  'lmm' is an integer giving the number of BFGS updates retained in
          the '"L-BFGS-B"' method, It defaults to '5'.
	 */
	int lmm = 5;
	/*
	 nbd is an integer array of dimension n.
	 On entry nbd represents the type of bounds imposed on the
	   variables, and must be specified as follows:
	   nbd(i)=0 if x(i) is unbounded,
		  1 if x(i) has only a lower bound,
		  2 if x(i) has both lower and upper bounds, and
		  3 if x(i) has only an upper bound.
	 On exit nbd is unchanged.

	*/
	int nbd = 0;
	if (lb != 0){
		if (ub != 0){
			nbd = 2;
		} else {
			nbd = 1;
		}
	} else if (ub != 0){
		nbd = 3;
	}
	
	
	/* did it fail? */
	int fail = 0;
	/*
	 * 'factr' controls the convergence of the '"L-BFGS-B"' method.
      Convergence occurs when the reduction in the objective is
      within this factor of the machine tolerance. Default is
      '1e7', that is a tolerance of about '1e-8'.
   */
	double factr = 1e+07;
	/*
	 * 'pgtol' helps control the convergence of the '"L-BFGS-B"' method.
      It is a tolerance on the projected gradient in the current
      search direction. This defaults to zero, when the check is
      suppressed.
   */
	double pgtol =  0;
	/*
	 * 'trace' Non-negative integer. If positive, tracing information on
      the progress of the optimization is produced. Higher values
      may produce more tracing information: for method '"L-BFGS-B"'
      there are six levels of tracing.  (To understand exactly what
      these do see the source code: higher levels give more
      detail.)
   */
   int trace = 0;
   //count how many times the optimization function was called
   int fncount = 0;
   //count how many times the gradient of the optimization function was called
	int grcount = 0;
	//maximum allowed number of iterations
	int maxit = 100;
	/*
	 did't quite get what the 'msg' argument is, it could be this:
	 'task' is a working string of characters of length 60 indicating
	 the current job when entering and quitting this subroutine.
	 */
	char msg[60];
	/*
	 'REPORT' The frequency of reports for the '"BFGS"', '"L-BFGS-B"'
	 and '"SANN"' methods if 'control$trace' is positive. Defaults
	 to every 10 iterations for '"BFGS"' and '"L-BFGS-B"', or
	 every 100 temperatures for '"SANN"'.
	 */
	int nREPORT = 10;
	
	//finally call the optimization function
	lbfgsb(n, lmm, x, lb, ub, &nbd, fx, fn, gr, &fail, ex, factr,
			pgtol, &fncount, &grcount, maxit, msg, trace, nREPORT);
	
	
	/*sometime it fails... nothing to do about it....
	if (fail){
		throw std::invalid_argument("The L-BFGS-B algorithm failed with this error message:\n" + std::string(msg));
	}
	*/
}

/*
static inline void bfgs_wrapper(optimfn fn, optimgr gr, void* ex, double* x, double* fx){
	int n = 1; 
	int trace = 0;
	//count how many times the optimization function was called
	int fncount = 0;
	//count how many times the gradient of the optimization function was called
	int grcount = 0;
	//maximum allowed number of iterations
	int maxit = 100;
	int nREPORT = 10;
	//R's defaults 
	double reltol = sqrt(DBL_EPSILON);
	double abstol = -INFINITY;
	//That's what R does in optim.c
	int mask = 1;
	// did it fail? 
	int fail = 0;
	
	vmmin(n, x, fx, fn, gr, maxit, trace, &mask, abstol, reltol, nREPORT, ex, &fncount, &grcount, &fail);
	
	if (fail){
		throw std::invalid_argument("The BFGS algorithm did not converge within the given number of iterations");
	}
}
*/
