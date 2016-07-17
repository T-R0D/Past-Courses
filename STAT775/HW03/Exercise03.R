setwd("C:/Users/Terence/Documents/GitHub/STAT775/HW06")

DATA.PATH <- "../DataSets/zip.data/"
ZIP.TRAIN.FILE.NAME <- paste0(DATA.PATH, "zip.train")
ZIP.TEST.FILE.NAME <- paste0(DATA.PATH, "zip.test")


read.data.tuples <- function(file.path.name) {
  data.fram.e <- read.table(file.path.name)

  data <- data.matrix(data.fram.e[, -1])

  data.tuple <- list(
    observations = data,
    labels = data.matrix(data.fram.e[, 1])
  )
  return(data.tuple)
}

#
# PCA
#

get.pca.summary <- function(data, num.components = 20) {
  #
  # Args:
  #   data: an n x m matrix of n observations of m dimensions
  #   num.components: the number of principle components to keep

  num.component.s <- min(ncol(data), num.components)
  n.obs <- nrow(data)
  full.dimensionality <- ncol(data)
  mu <- colMeans(data)

  # get centered data
  x <- data
  for (i in 1:n.obs) {
    x[i, ] <- data[i, ] - mu
  }

  # covariance matrix
  sigma <- t(x) %*% x
  sigma <- sigma * (1.0 / n.obs)

  eigen.decomposition <- eigen(sigma, F)  # TODO: is cov symmetric? Not sure...
  eigen.vectors <- eigen.decomposition$vectors[, 1:num.component.s]

  pca.summary <- list(
    rotation = eigen.vectors,
    mu = matrix(mu, nrow = 1, ncol = full.dimensionality)
  )

  return(pca.summary)
}

predict <- function(pca.summary, data) {
  #
  # Args:
  #   data: an n x d matrix of observations; rows are observations
  #   pca.summary: a tuple of the rotation matrix (eigenvectors as columns) and
  #                the mean (row vector of column means) computed in the pca
  #                computations

  x <- data
  for (i in 1:nrow(data)) {
    x[i, ] <- data[i, ] - pca.summary$mu
  }

  return (t(t(pca.summary$rotation) %*% t(x)))
}


train <- read.data.tuples(ZIP.TRAIN.FILE.NAME)

pca.model <- get.pca.summary(train$observations, 16)

train$observations <- predict(pca.model, train$observations)

train.frame <- data.frame('class' = train$labels, train$observations)

CLASSES <- list(0,1,2,3,4,5,6,7,8,9)
class.summaries <- list(10)
for (k in CLASSES) {
  data.subset <- subset(train.frame, class == k)
  n <- nrow(data.subset)
  mu <- colMeans(data.subset[, -1])
  sigma <- cov(data.subset[, -1])
  class.summaries[[k + 1]] <- list(
    label = k,
    n = n,
    mu = mu,
    sigma = sigma
  )
}

gaussian.pdf <- function(k, x) {
  x <- matrix(x, ncol = 1)
  scale.f <- 1.0 / sqrt(((2.0 * pi) ^ 16) * det(k$sigma))
  difference <- x - matrix(k$mu, ncol = 1)
  exponent <- -0.5 * (t(difference) %*% solve(k$sigma) %*% difference)
  return(scale.f * exp(exponent))
}

classify.object <- function(classes, x) {
  highest.prob <- 0.0
  most.probable.class.label <- 0
  for (i in 1:length(classes)) {
    prob <- gaussian.pdf(k = classes[[i]], x = x) * classes[[i]]$n
    if (prob > highest.prob) {
      highest.prob <- prob
      most.probable.class.label <- classes[[i]]$label
    }
  }

  return(most.probable.class.label)
}

test <- read.data.tuples(ZIP.TEST.FILE.NAME)
test$observations <- as.matrix(predict(pca.model, test$observations))
test$observations <- matrix(
  test$observations[, 1:16],
  nrow = nrow(test$observations),
  ncol = 16
)

num.correct <- 0
confusion.matrix <- matrix(0, nrow = 10, ncol = 11)
for (i in 1:nrow(test$labels)) {
  prediction <- classify.object(class.summaries, x = test$observations[i, ])
  if (prediction == test$labels[[i]]) {
    num.correct <- num.correct + 1
  }

  confusion.matrix[test$labels[[i]] + 1, prediction + 1] <-
    confusion.matrix[test$labels[[i]] + 1, prediction + 1] + 1

}

totals <- rowSums(confusion.matrix)
for (i in 1:nrow(confusion.matrix)) {
  confusion.matrix[i, 11] <- (confusion.matrix[i,i] / totals[[i]]) * 100
}


print(100 * num.correct / nrow(test$labels))
print(confusion.matrix)
