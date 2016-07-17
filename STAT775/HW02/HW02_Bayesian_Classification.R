##
# HW02: Bayesian Classification
#
##

######################
# some initial setup #
######################
working.directory = "~/Desktop/STAT775/HW02/"
train.file.name = "data/zip.train"
test.file.name = "data/zip.test"

setwd(working.directory)

require(caret)  # for confusion matrix
# require(ggplot2)
# require(reshape)


#################################
# Data Preparation and Handling #
#################################
create.test.point.struct <- function(labeled.data.row) {
  return(
    list(
      actual.class = as.numeric(labeled.data.row[1]),
      classified.as = -1,
      value = t(data.matrix(labeled.data.row[-1]))
    )
  )
}


zip.train <- read.table(
  train.file.name,
  header = F,
  row.name = NULL
)
zip.test <- read.table(
  test.file.name,
  header = F,
  row.name = NULL
)


test.point.structs <- list()
for (i in 1:nrow(zip.test)) {
  test.point.structs[[i]] <- create.test.point.struct(zip.test[i, ])
}


train.factor <- factor(zip.train[, 1])
test.factor <- factor(zip.test[, 1])


########################
# 'Train' the model(s) #
########################
create.class.summary <- function(labeled.data.matrix, class) {
  data <- data.matrix(subset(labeled.data.matrix, V1 == class)[, -1])
  
  covariance.matrix <- (cov(data) + diag(x = 0.7, ncol(data)))
#   if (det(covariance.matrix) - 0.1 <= 0) {
#     covariance.matrix <- covariance.matrix +
#                          (diag(x = 0.0001, nrow(covariance.matrix)))
#   }
  
  return(
    list(
      label = as.numeric(labeled.data.matrix[1, 1]),
      prior = nrow(data) / nrow(labeled.data.matrix),
      mean = data.matrix(colMeans(data)),
      covariance = cov(data)  # cov wants data by rows, result is symmetric
    )
  )
}

bayesian.digit.classifiers <- list()
for (level in levels(train.factor)) {
  bayesian.digit.classifiers[[level]] <- create.class.summary(zip.train, level) 
}


######################################
# Perform Classification on Test Set #
######################################
multi.gaussian.pdf <- function(x, mu, sigma) {
  # returns the probability of a multivariate Gaussian
  # defined by the mean and covariance parameters
  #
  # Args:
  #   x: The test point/vector to test (n x 1).
  #   mu: A column vector of the distribution's mean (n x 1).
  #   sigma: The covariance matrix of the distribution (n x n).
  #
  # Returns:
  #   probability: The relative frequency/likelihood of
  #                observing x in this distribution (scalar)
  #
  sigma.adj <- NULL
  if (det(sigma) - 0.0000001 <= 0) {
    print("adjusting")
    sigma.adj <- sigma + diag(x=0.7, nrow(sigma))    
  } else {
    sigma.adj <- sigma
  }
  if (det(sigma.adj) - 0.0000001 <= 0) {
    print("what the fuck am I supposed to do?")
    solve(sigma.adj)
    print("I mean, this is impossible!")
  }
  
  
  k <- nrow(sigma.adj)
  scaling.factor <- 1.0 / sqrt((2 * pi)^(k) * det(sigma.adj))
  exponent <- as.numeric(-1/2 * (t(x - mu) %*% solve(sigma.adj) %*% (x - mu)))
  print(sprintf("%g * exp(%g)", scaling.factor, exponent))
  return(as.numeric(scaling.factor * exp(exponent)))
}

classify.object <- function(feature.vector, classifiers) {
  probability <- list()
  class <- list()

  for (i in 1:length(classifiers)) {
    class[[i]] <- classifiers[[i]]$label
    probability[[i]] <- multi.gaussian.pdf(
      x = feature.vector,
      mu = classifiers[[i]]$mean,
      sigma = classifiers[[i]]$covariance
    ) * classifiers[[i]]$prior
  }
  
  print("list-class:")
  print(typeof(class[[1]]))
  print(class)
  
  class.probs <- data.frame(probability, as.factor(class))
  
  str(class.probs)
  
  class.probs <- class.probs[order(-(class.probs$probability)), ]
  return(class.probs$class[[1]])
}


# for (i in 1:length(test.point.structs)) {
#   test.point.structs[[i]]$classified.as <- classify.object(
#     test.point.structs[[i]]$value,
#     bayesian.digit.classifiers
#   )
# }


############
# Plotting #
############
library(caret)

results <-data.matrix(read.table("results"))
dimnames(results) <- list(c(0,1,2,3,4,5,6,7,8,9), c(0,1,2,3,4,5,6,7,8,9))

actuals.f <- list()
predictions.f <- list()
for (i in 1:nrow(results)) {
  for (j in 1:ncol(results)) {
    actuals.f <- c(actuals.f, rep(i - 1, results[i, j]))
    predictions.f <- c(predictions.f, rep(j - 1, results[i, j]))
  }
}
actuals.f <- factor(unlist(actuals.f))
predictions.f <- factor(unlist(predictions.f))

confusionMatrix(data = predictions.f, reference = actuals.f)


