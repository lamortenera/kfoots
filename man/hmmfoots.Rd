\name{hmmfoots}
\alias{hmmfoots}
\title{Fit an hmm}
\usage{
  hmmfoots(counts, models, trans = NA, tol = 1e-08,
    maxiter = 100, output = FALSE, seqlens = ncol(counts))
}
\arguments{
  \item{counts}{matrix of non-negative integers. Columns
  represent datapoints and rows dimensions}

  \item{models}{either the desired number of cluster, or a
  specific initial value for the models. See the item
  \code{models} in the return values to see how the
  parameters are specified}

  \item{trans}{initial value for the transition
  probabilities. If not provided equally likely transitions
  and fully connected markov model will be assumed.}

  \item{tol}{error tolerance used when checking whether the
  parameters have changed from one iteration to the next}

  \item{maxiter}{maximum number of iterations in the EM
  algorithm}

  \item{output}{print some output during execution}

  \item{seqlens}{print some output during execution}
}
\value{
  a list with the parameters of the fitted model:
  \item{models}{a list containing the parameters of each
  component.  These are specified in another list with
  items \code{mu}, \code{r} and \code{ps}. \code{mu} and
  \code{r} correspond to parameters \code{mu} and
  \code{size} in the R-function \code{\link{dnbinom}}.  Ps
  specifies the parameters of the multinomial and they sum
  up to 1.} \item{mix_coeff}{the mixture coefficients of
  the mixture model} \item{loglik}{the log-likelihood of
  the whole dataset.} \item{posteriors}{A matrix of size
  \code{length(models)*ncol(counts)} containing the
  posterior probability that a given datapoint is generated
  by the given mixture component}
  \item{converged}{\code{TRUE} if the algorithm converged
  in the given number of iterations, \code{FALSE}
  otherwise} \item{llhistory}{time series containing the
  log-likelihood of the whole dataset across iterations}
  \item{viterbi}{see output of \code{viterbi}}
}
\description{
  Fit to some count data an hmm where the emission
  probabilities are modelled as negative multinomials
}

