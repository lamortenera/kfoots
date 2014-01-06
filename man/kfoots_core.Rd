\name{kfoots_core}
\alias{kfoots_core}
\title{Fit mixture model}
\usage{
  kfoots_core(counts, k, mix_coeff = NULL, tol = 1e-08,
    maxiter = 100, nthreads = 1, verbose = FALSE)
}
\arguments{
  \item{counts}{matrix of non-negative integers. Columns
  represent datapoints and rows dimensions}

  \item{k}{either the desired number of cluster, or a
  specific initial value for the models. See the item
  \code{models} in the return values to see how the model
  parameters should be formatted}

  \item{mix_coeff}{initial value for the mixture
  coefficients. If not provided equally likely clusters
  will be assumed at the first iteration.}

  \item{tol}{error tolerance used when checking whether the
  parameters have changed from one iteration to the next}

  \item{maxiter}{maximum number of iterations in the EM
  algorithm}

  \item{verbose}{print some output during execution}
}
\value{
  a list with the parameters of the fitted model:
  \item{models}{a list containing the parameters of each
  model.  These are specified in another list with items
  \code{mu}, \code{r} and \code{ps}. \code{mu} and \code{r}
  correspond to parameters \code{mu} and \code{size} in the
  R-function \code{\link{dnbinom}}.  Ps specifies the
  parameters of the multinomial and they sum up to 1.}
  \item{mix_coeff}{the mixture coefficients of the mixture
  model} \item{loglik}{the log-likelihood of the whole
  dataset.} \item{posteriors}{A matrix of size
  \code{length(models)*ncol(counts)} containing the
  posterior probability that a given datapoint is generated
  by the given mixture component}
  \item{converged}{\code{TRUE} if the algorithm converged
  in the given number of iterations, \code{FALSE}
  otherwise} \item{llhistory}{time series containing the
  log-likelihood of the whole dataset across iterations}
}
\description{
  Fit a mixture of negative multinomials to some count data
}

