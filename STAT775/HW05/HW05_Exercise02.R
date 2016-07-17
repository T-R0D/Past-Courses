##############
# STAT775: Machine Learning
#
# HW05
# Exercise 02
#
# Using the zip code digit data from the ESL website, use Single Neuron
# Neural Networks to perform binary classification between 2s and 3s, 4s and 5s,
# and 7s and 9s.
###############

#
# Initial Setup
#
setwd("C:/Users/Terence/Documents/GitHub/STAT775/HW05")

#
# Data Cleaning
#
DATA.PATH <- "../DataSets/zip.data/"
ZIP.TRAIN.FILE.NAME <- paste0(DATA.PATH, "zip.train")
ZIP.TEST.FILE.NAME <- paste0(DATA.PATH, "zip.test")

create.data.tuple <- function(file.path.name, class0, class1) {
  data <- read.table(file.path.name)
  data <- subset(data, V1 == class0 | V1 == class1)

  labels <- matrix(0, nrow(data), 1)
  for (i in 1:nrow(data)) {
    if (data[i, 1] == class1) {
      labels[i, 1] <- 1.0
    }
  }

  data <- data[, -1]

  data.tuple <- list(
    labels = data.matrix(labels),
    observations = data.matrix(data)
  )
  return(data.tuple)
}

#
# Neuron Structure
#
create.neuron <- function(dim, activation.f, activation.f.derivative) {
#
# Args:
#   dim: an integer dimensionality of the inputs (do not include bias)
#   activation.f: the activation function of the neuron
#   activation.f.derivative: a the function that is the derivative of activation.f

  w <- matrix(rand(), dim + 1, 1)

  return(list(
    w = w,
    f = activation.f,
    f.prime = activation.f.derivative,
    output = 0,
    error.prime = 9999
  ))
}


sigmoid.f <- function(x) {
  return(1.0 / (1.0 + exp(-x)))
}

sigmoid.f.prime <- function(x) {
  return(sigmoid.f(x) * (1.0 - sigmoid.f(x)))
}

weighted.sum.inputs <- function(b, x) {
#
# Args:
#   x: a p x 1 column vector of feature values
#   b: a (p + 1) x 1 column vector of input weights
  return((t(b[-1, ]) %*% x) - b[1,1])
}


#
#
#

test.cases <- list(
  list(2, 3),
  list(4, 5),
  list(7, 9)
)

case <- test.cases[[1]]
i <- 1
learn.rate <- 0.1

for (case in test.cases) {
  train.data <- create.data.tuple(ZIP.TRAIN.FILE.NAME, case[[1]], case[[2]])
  W <- matrix(runif(ncol(train.data$observations) + 1))

  k <- 1000
  while (k > 0) {
    for (i in 1:nrow(train.data$labels)) {
      x <- data.matrix(train.data$observations[i, ])
      y <- train.data$labels[[i]]
      input <- weighted.sum.inputs(x = x, b = W)
      o <- sigmoid.f(input)
      e <- y - o
      W <- W + learn.rate * (rep(((o * (1.0 - o)) * e), 257) * rbind(1.0, x))




#       D <- sigmoid.f.prime(input) * e
#       W <- W - rep((learn.rate * o * D), 257)
    }

    k <- k - 1
  }

  test.data <- create.data.tuple(ZIP.TEST.FILE.NAME, case[[1]], case[[2]])
  confusion.matrix <- matrix(0, 2, 2 + 1)
  rownames(confusion.matrix) <- c(case[[1]], case[[2]])
  colnames(confusion.matrix) <- c(case[[1]], case[[2]], '%.Correct')
  correct <- 0

  for (i in 1:nrow(test.data$labels)) {
    x <- data.matrix(test.data$observations[i, ])
    prediction <- round(sigmoid.f(weighted.sum.inputs(x = x, b = W)))

    if (test.data$labels[[i]] == 0) {
      if (prediction == 0) {
        confusion.matrix[1, 1] <- confusion.matrix[1, 1] + 1
      } else {
        confusion.matrix[1, 2] <- confusion.matrix[1, 2] + 1
      }
    } else {
      if (prediction == 0) {
        confusion.matrix[2, 1] <- confusion.matrix[2, 1] + 1
      } else {
        confusion.matrix[2, 2] <- confusion.matrix[2, 2] + 1
      }
    }

    if (prediction == test.data$labels[[i]]) {
      correct <- correct + 1
    }
  }
  print(correct / nrow(test.data$labels))
  confusion.matrix[1, 3] <- confusion.matrix[1, 1] /
                            (confusion.matrix[1, 1] + confusion.matrix[1, 2])
  confusion.matrix[2, 3] <- confusion.matrix[2, 2] /
                            (confusion.matrix[2, 1] + confusion.matrix[2, 2])
  print(confusion.matrix)
}

