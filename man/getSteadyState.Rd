\name{getSteadyState}
\alias{getSteadyState}
\title{Get a steady state of a transition matrix.}
\usage{
getSteadyState(trans)
}
\arguments{
  \item{trans}{transition matrix (rows are previous state,
  columns are next state)}
}
\value{
a vector with a steady state distribution
}
\description{
It should give a similar result as \code{rep(1/ncol(trans),
ncol(trans)) trans^(big number)} except that oscillating
behaviours are averaged out.
}

