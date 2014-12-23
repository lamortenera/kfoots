\name{forward_backward}
\alias{forward_backward}
\title{Forward-backward algorithm}
\usage{
forward_backward(initP, trans, lliks, seqlens, posteriors, nthreads = 1L)
}
\arguments{
  \item{initP}{matrix of initial probabilities: each column
  corresponds to a sequence}

  \item{trans}{transition matrix (rows are previous state,
  columns are next state)}

  \item{lliks}{matrix with emission probabilities for each
  datapoint and each state. Columns are datapoints and rows
  are states.}

  \item{seqlens}{length of each subsequence of datapoints
  (set this to ncol(lliks) if there is only one sequence).}
}
\value{
a list with the following arguments:
\item{posteriors}{posterior probability of being in a
certain state for a certain datapoint}
\item{tot_llik}{total log-likelihood of the data given the
hmm model} \item{new_trans}{update for the transition
probabilities (it is already normalized)}
}
\description{
Forward-backward algorithm using the scaling technique.
That's more stable (and maybe even faster) than the method
with the logarithm. Warning: this function overwrites the
lliks matrix. This is probably a bad idea because normally
in the last loop you want to use the same matrix for
forward_backward and viterbi. I might change that in the
future.
}

