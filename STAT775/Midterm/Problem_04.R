##################
# STAT775
# Midterm
# Problem 04
#
#
##################


#
# Initial Setup
#
setwd("C:/Users/Terence/Documents/GitHub/STAT775/HW03/")
DATA.FILE.NAME <- "../DataSets/prostate_cancer/prostate_cancer_data"

#
# Data Handling
#
# these include the intercept that gets added to the matrices
FEATURE.INDICES <- 1:9
TARGET.INDEX <- 10
TRAIN.SET.FLAG <- 11

scale.outputs <- function(x) {
  # Args:
  #  x: a numerical vector, column or row

  range <- max(x) - min(x)
  scaled.x <- (x - min(x)) / range
  return(scaled.x)
}

unscale.outputs <- function(x, max.unscaled, min.unscaled) {
  range <- max.unscaled - min.unscaled
  return((x * range) + min.unscaled)
}


data <- read.table(DATA.FILE.NAME)
# according to the dataset info, this needs to be done
data[, 1:8] <- scale(data[, 1:8], T, T)
intercept <- rep(1, nrow(data))
data <- data.frame(Intercept = intercept, data)

# store data with rows as observations
train.data <- data.matrix(subset(data, data$train == T))[, FEATURE.INDICES]
train.targets <- data.matrix(subset(data, data$train == T))[, TARGET.INDEX]

test.data <- data.matrix(subset(data, data$train == F))[, FEATURE.INDICES]
test.targets <- data.matrix(subset(data, data$train == F))[, TARGET.INDEX]

max.test.target <- max(test.targets)
min.test.target <- min(test.targets)
unscaled.test.targets <- test.targets

train.targets <- scale.outputs(train.targets)
test.targets <- scale.outputs(test.targets)

#
# Regression
#
ols.regression <- function(x, y) {
  #
  # Args:
  #   x: a matrix of observations; each observation is stored as a row
  #   y: a column vector of the targets, or actual value/label, for each
  #      observation

  return(list(
    beta.hat = solve(t(x) %*% x) %*% t(x) %*% matrix(y, ncol = 1)
  ))
}

ols.predict <- function(model, x) {
  #
  # Args:
  #   model: an n x 1 column vector of regression coefficients
  #   x: a ? x n matrix of observation data

  return(t(model$beta.hat) %*% t(x))
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
ols.model <- ols.regression(x = train.data, y = train.targets)
predictions <- ols.predict(model = ols.model, x = test.data)
OLS.Predictions <- as.list(predictions)
# OLS.Predictions <- as.list(unscale.outputs(predictions, max.test.target, min.test.target))

ols.RSS <- sum((test.targets - predictions) ^ 2)
print(ols.RSS)


# Neural Net
NUM.EPOCHS <- 1000
prostate.net <- construct.neural.net(
  topology = c(ncol(train.data), 10, 1),
  activation = sigmoid,
  activation.deriv = sigmoid.derivative
)

for (t in 1:NUM.EPOCHS) {
  for (i in 1:length(train.targets)) {
    prostate.net <- apply.inputs(
      net = prostate.net,
      x = matrix(train.data[i, ], nrow = 1)
    )
    prostate.net <- backprop.weight.update(
      net = prostate.net,
      target = matrix(train.targets[[i]], ncol = 1),
      learning.rate = 0.2
    )
  }
}

net.RSS <- 0
net.predictions <- list()
for (i in 1:length(test.targets)) {
  prostate.net <- apply.inputs(
    net = prostate.net,
    x = matrix(test.data[i, ], nrow = 1)
  )

  net.predictions[[i]] <- prostate.net$output

  error.squared <- (prostate.net$output - test.targets[[i]]) ^ 2

  net.RSS <- net.RSS + error.squared
}
NET.Predictions <- as.list(net.predictions)
# NET.Predictions <- as.list(unscale.outputs(unlist(net.predictions), max.test.target, min.test.target))

unscaled.test.targets <- test.targets

ols.error <- list()
net.error <- list()
for (i in 1:length(unscaled.test.targets)) {
  ols.error[[i]] <- unscaled.test.targets[[i]] - OLS.Predictions[[i]]
  net.error[[i]] <- unscaled.test.targets[[i]] - NET.Predictions[[i]]
}

results <- data.frame(
  cbind(
    unscaled.test.targets,
    OLS.Predictions,
    NET.Predictions,
    ols.error,
    net.error
  )
)

print(data.frame('OLS RSS' = ols.RSS, 'Neural Net RSS' = net.RSS))
print(results)
