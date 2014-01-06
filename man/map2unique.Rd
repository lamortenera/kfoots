\name{map2unique}
\alias{map2unique}
\title{Group unique values of a vector}
\usage{
  map2unique(v)
}
\arguments{
  \item{v}{a vector}
}
\value{
  a list with the following items: \item{values}{unique and
  sorted values of \code{v}} \item{map}{a vector such that
  \code{v[i] = values[map[i]]} for every i}
}
\description{
  Group unique values of a vector
}

