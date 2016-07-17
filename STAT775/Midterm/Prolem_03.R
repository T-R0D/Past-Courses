##################
# STAT775
# Midterm
# Problem 03
#
#
##################


#
# Initial Setup
#
setwd("C:/Users/Terence/Documents/GitHub/STAT775/Midterm/")
# setwd("~/STAT775/HW03/")

DATA.FILE.NAME <- "../DataSets/south.african.heart.disease/data.disease"


#
# Data Handling
#
shuffle.data <- function(x) {
  # x: a numeric matrix

  for (i in 1:nrow(x)) {
    one <- round(runif(1, 1, nrow(x)))
    other <- round(runif(1, 1, nrow(x)))

    temp <- x[one, ]
    x[one, ] <- x[other, ]
    x[other, ] <- temp
  }

  return(x)
}

code.data <- function(dat.a) {
  family.history <- rep(0.0, nrow(dat.a))

  data <- data.frame(dat.a[, 1:4], family.history, dat.a[6:10])
#   data <- data.frame(dat.a[, 'tobacco'], dat.a[, 'ldl'], family.history, dat.a[, 'age'], dat.a[, 'chd'])

  for (i in 1:nrow(data)) {
    if (dat.a[i, 'famhist'] == 'Present') {
      data[i, 'family.history'] <- 1.0
    }
  }

  return(shuffle.data(data.matrix(data)))
}

get.data.tuples <- function(data) {
  last.column <- ncol(data)

  return(list(
    observations = (data.matrix(data[, -last.column])),
    targets = data.matrix(data[, last.column])
  ))
}


#
# Logistic Regression
#
logistic.predict <- function(x, b) {
  #
  # Args:
  #   x: a p x 1 column vector of observation values
  #   b: a p X 1 column vector of model weights (betas)

  e <- exp(matrix(b, nrow = 1) %*% rbind(1.0, matrix(x, ncol = 1)))
  return(e / (1.0 + e))
}

newton.raphson.iteration <- function(X, y, old.betas, step.size = 0.5) {
  #
  # Args:
  #   X: N x p matrix of observations
  #   y: column vector of N target labels
  #   old.betas: column vector of (p + 1) model parameters
  #   step.size: a rate controlling parameter

  p <- matrix(0, nrow = nrow(y), ncol = 1)
  for (i in 1:nrow(y)) {
    p[i, 1] <- logistic.predict(x = data.matrix(X[i, ]), b = old.betas)
  }

  W <- diag(1, nrow(y), nrow(y))
  for (i in 1:nrow(y)) {
    W[i, i] <- p[i, 1] * (1.0 - p[i, 1])
  }

  X.b <- cbind(1.0, X)  # X with 1s prepended for bias term

  z <- (X.b %*% old.betas) +
    (solve(W + diag(0.001, nrow(W), ncol(W))) %*% (y - p))

  step <- solve((t(X.b) %*% W %*% X.b) + diag(0.001, ncol(X.b), ncol(X.b))) %*%
    t(X.b) %*% (y - p)
  step <- step * step.size

  return(old.betas + step)
}

train.logistic.model <- function(X, y, n.iterations = 400, step.size = 0.2) {
  #
  # Args:
  #   X: N x p matrix of observations, observations are rows
  #   y: N x 1 column vector of class labels {0, 1}
  #   n.iterations: the number of approximation iterations to perform
  #   step.size: used to slow the rate of convergence to prevent overshooting

  # start with model parameters at 0 - arbitrary starting point
  betas <- matrix(0, nrow = ncol(X) + 1, 1)

  # iterate until parameter updates become arbitrarily small
  old.betas <- betas + 1.0
  i <- 0
  while (i < n.iterations) {
    betas <- newton.raphson.iteration(
      X = X,
      y = y,
      old.betas = betas,
      step.size = step.size
    )

    # TODO: use method for finding convergence, not just arbitrary iterations

    i <- i + 1
  }

  return(betas)
}


#
# Neural Net
#

sigmoid <- function(x) {
  #
  # Args:
  #   x: a numeric or vector; the function is applied element-wise

  return(1.0 / (1.0 + exp(-x)))
}

sigmoid.derivative <- function(x) {
  #
  # Args:
  #   x: a numeric or vector

  return(sigmoid(x) * (1.0 - sigmoid(x)))
}

construct.neural.net <- function(
  topology = c(2, 2, 1),
  activation = sigmoid,
  activation.derivative = sigmoid.derivative,
  debug = F) {
  #
  # Args:
  #   topology: a list or vector of the dimensions of each layer
  #   activation: a function to be used for the activation of each unit
  #   activation.derivative: a function that is the derivative of activation

  layer.weights <- list()
  derivative.matrices <- list()
  outputs <- list()

  previous.layer.dim <- 1
  next.layer.dim <- 1
  for (i in 1:(length(topology) - 1)) {
    previous.layer.dim <- topology[[i]] + 1  # +1 for bias
    next.layer.dim <- topology[[i + 1]]
    num.elements <- (previous.layer.dim) * next.layer.dim

    layer.weights[[i]] <- matrix(
      if(debug) {rep(1, num.elements)}
      else {runif(n = num.elements, min = -0.001, max = 0.001)},
      nrow = previous.layer.dim,
      ncol = next.layer.dim
    )

    outputs[[i]] <- matrix(0, nrow = next.layer.dim, ncol = 1)
    derivative.matrices[[i]] <- diag(0, next.layer.dim)
  }

  return(list(
    input.dim = topology[[1]],
    output.dim = next.layer.dim,  # should be dim of last layer
    n.layers = length(layer.weights),
    activation = activation,
    activation.deriv = activation.derivative,
    input = matrix(0, nrow = 1, ncol = topology[[1]]),
    output = matrix(0, nrow = tail(topology, 1)[[1]], 1),
    weights = layer.weights,
    outputs = outputs,
    derivatives = derivative.matrices
  ))
}

apply.inputs <- function(net, x, for.training = T) {
  #
  # Args:
  #   x: a 1 x n vector of inputs; n should be the same as for net
  #   net: a structure with all of the appropriate data for a neural network,
  #        as created by construct.neural.net()
  #   for.training[T]: currently unused

  net$input <- matrix(x, nrow = 1)
  previous.output <- cbind(net$input, 1)
  for (i in 1:net$n.layers) {
    weighted.sums <- previous.output %*% net$weights[[i]]

    net$outputs[[i]] <- net$activation(weighted.sums)

    net$derivatives[[i]] <- diag(
      as.list(net$activation.deriv(weighted.sums)),
      length(net$outputs[[i]])
    )

    previous.output <- cbind(net$outputs[[i]], 1)
  }

  net$output <- t(tail(net$outputs, 1)[[1]])

  return(net)
}

backprop.weight.update <- function(net, target, learning.rate = 2) {
  #
  # Args:
  #   net: a neural net object that has had inputs applied and derivatives stored
  #   target: a column vector; the target output that should have been observed
  #   learning.rate: the learning rate of the network
  #                  TODO: refactor learning.rate to be less hacky, allow for
  #                        advanced techniques

  last.index <- net$n.layers

  deltas <- list()
  error <- matrix(net$output, ncol = 1) - matrix(target, ncol = 1)
  deltas[[last.index + 1]] <- error
  W <- diag(1, nrow = nrow(error))
  D <- net$derivatives[[last.index]]
  for (i in last.index:1) {
    deltas[[i]] <- D %*% W %*% deltas[[i + 1]]

    if (i > 1) {
      D <- net$derivatives[[i - 1]]
      W <- net$weights[[i]]
      W <- W[1:(nrow(W) - 1), ]
    }
  }

  weight.updates <- list()
  o.hat <- cbind(net$input, 1)
  for (i in 1:last.index) {
    weight.updates[[i]] <- -learning.rate * t(deltas[[i]] %*% o.hat)

    if (i < last.index) {
      o.hat <- cbind(net$outputs[[i]], 1)
    }
  }

  for (i in 1:length(weight.updates)) {
    net$weights[[i]] <- net$weights[[i]] + weight.updates[[i]]
  }

  return(net)
}


######
# MAIN
######
data.fram.e <- read.table(
  "http://www-stat.stanford.edu/~tibs/ElemStatLearn/datasets/SAheart.data",
  sep = ",",
  head = T,
  row.names = 1
)
data <- code.data(data.fram.e)
boundary <- 2 * nrow(data) / 3
train <- get.data.tuples(data[1:boundary, ])
test <- get.data.tuples(data[-(1:boundary), ])

# Logistic
logistic.model <- train.logistic.model(X = train$observations, y = train$targets)
print(logistic.model)

glm.out <- glm(train$targets ~ ., family=binomial(logit), data=data.frame(train$observations))

# logistic.model <- matrix(c(-4.204, 0.081, 0.168, 0.924, 0.044), ncol = 1)


logistic.confusion <- matrix(0, nrow = 3, ncol = 3)
for (i in 1:nrow(test$targets)) {
  prediction <- round(logistic.predict(x = test$observations[i, ], b = logistic.model))
  label <- test$targets[[i]]

  logistic.confusion[label + 1, prediction + 1] <-
    logistic.confusion[label + 1, prediction + 1] + 1
}


# Neural Net
NUM.EPOCHS <- 700
chd.net <- construct.neural.net(
  topology = c(9, 5, 1),
  activation = sigmoid,
  activation.deriv = sigmoid.derivative
)

for (t in 1:NUM.EPOCHS) {
  for (i in 1:nrow(train$targets)) {
    chd.net <- apply.inputs(
      net = chd.net,
      x = matrix(train$observations[i, ], nrow = 1)
    )
    chd.net <- backprop.weight.update(
      net = chd.net,
      target = matrix(train$targets[i, ], ncol = 1),
      learning.rate = 1.2
    )
  }
}

net.confusion <- matrix(0, 3, 3)
for (i in 1:nrow(test$observations)) {
  prediction <- round(
    apply.inputs(
      net = chd.net,
      x = matrix(test$observations[i, ], nrow = 1),
      for.training = F
    )$output
  )

  label <- test$targets[[i]]

  net.confusion[label + 1, prediction + 1] <-
    net.confusion[label + 1, prediction + 1] + 1
}


# library('nnet')
# b.net <- nnet(x = train$observations, y = train$targets, size = 3, maxit = 1000)
# pr <- round(predict(b.net, test$observations))
# net.confusion <- matrix(0, 3, 3)
# for (i in 1:nrow(pr)) {
#   prediction <- pr[[1]]
#   label <- test$targets[[i]]
#
#   net.confusion[label + 1, prediction + 1] <-
#     net.confusion[label + 1, prediction + 1] + 1
# }



# Results
compute.confusion.percentages <- function(confusion) {
  rownames(confusion) <- c('Actual No CHD', 'Actual CHD', '%.Correct')
  colnames(confusion) <- c('Actual No CHD', 'Actual CHD', '%.Correct')

  row.sums <- rowSums(confusion)
  col.sums <- colSums(confusion)
  correct <- 0
  for (i in 1:2) {
    confusion[i, 3] <- confusion[i, i] / row.sums[[i]]
    confusion[3, i] <- confusion[i, i] / col.sums[[i]]
    correct <- correct + confusion[i, i]
  }
  confusion[[3,3]] <- correct / sum(row.sums)

  return(confusion)
}

logistic.confusion <-compute.confusion.percentages(logistic.confusion)
net.confusion <-compute.confusion.percentages(net.confusion)
print(logistic.confusion, digits = 2)
print(net.confusion, digits = 2)

