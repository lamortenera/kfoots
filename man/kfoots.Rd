\name{kfoots}
\alias{kfoots}
\title{Wrapper around \code{\link{kfoots_core}}.}
\usage{
  kfoots(counts, k, nstart = 1, verbose = FALSE, cores = 1,
    ...)
}
\arguments{
  \item{counts}{see documentation for
  \code{\link{kfoots_core}}}

  \item{models}{either the desired number of cluster, or a
  specific initial value for the models. In the second case
  and if \code{nstart}>1, only the first iteration will use
  the specified initial parameters.}

  \item{nstart}{number of starting points}

  \item{output}{print some output during execution}

  \item{cores}{number of cores to be used. It is used to
  run the different initializations of \code{kfoots_core}
  in parallel, \code{kfoots_core} is not parallelized yet.
  Setting \code{cores > nstart} is the same as setting
  \code{cores = nstart}.}

  \item{...}{see documentation for
  \code{\link{kfoots_core}}}
}
\value{
  see documentation for \code{\link{kfoots_core}}. The
  followings are additional items added to the list:
  \item{converged_vs_total}{if \code{nstart} > 1, the total
  number of times the algorithm converged vs the number of
  inizialisations.}
}
\description{
  It implements multiple starting points and outer
  parallelization.
}

