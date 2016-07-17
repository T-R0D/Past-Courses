##############
# STAT775: Machine Learning
#
# HW05
# Exercise 01
#
# Using the zip code digit data from the ESL website, use Logistic Regression
# to perform binary classification between 2s and 3s, 4s and 5s,
# and 7s and 9s.
###############

#
# Initial Setup
#
setwd("C:/Users/Terence/Documents/GitHub/STAT775/HW07")

#
# Data Cleaning
#
DATA.PATH <- "../DataSets/zip.data/"
ZIP.TRAIN.FILE.NAME <- paste0(DATA.PATH, "zip.train")
ZIP.TEST.FILE.NAME <- paste0(DATA.PATH, "zip.test")

construct.classification.frame <- function(data, targets) {
  return(
    data.frame(
      'target' = targets,
      'prediction' = rep(0, length(targets)),
      data
    )
  )
}

get.data <- function(
  train.file, test.file, class.1, class.2, num.principle.components = 50) {
  # Args:
  #   train.file:
  #   test.file:
  #   class.1:
  #   class.2
  #   num.principle.components:
  #

  train <- read.table(train.file)
  train <- subset(train, V1 == class.1 | V1 == class.2)
  for (i in 1:nrow(train)) {
    train[i, 1] <- if (train[i, 1] == class.1) {1} else {2}
  }

  test <- read.table(test.file)
  test <- subset(test, V1 == class.1 | V1 == class.2)
  for (i in 1:nrow(test)) {
    test[i, 1] <- if (test[i, 1] == class.1) {1} else {2}
  }

  pca.model <- prcomp(train[, -1])

  train[, -1] <- predict(pca.model, train[, -1])
  train <- train[, 1:(num.principle.components + 1)]
  train <- construct.classification.frame(
    targets = train[, 1],
    data = train[, -1]
  )

  test[, -1] <- predict(pca.model, test[, -1])
  test <- test[, 1:(num.principle.components + 1)]
  test <- construct.classification.frame(
    targets = test[, 1],
    data = test[, -1]
  )

  return(list(
    train = train,
    test = test
  ))
}

#
# Data Presentation and Evaluation
#

compute.confusion.matrix <- function(predictions, targets, num.classes = 2) {
  confusion.matrix <- matrix(0, nrow = num.classes + 1, ncol = num.classes + 1)
  for (i in 1:length(predictions)) {
    confusion.matrix[targets[[i]], predictions[[i]]] <-
      confusion.matrix[targets[[i]], predictions[[i]]] + 1
  }
  confusion.matrix[num.classes + 1, num.classes + 1] <-
    sum(diag(confusion.matrix)) / sum(confusion.matrix)
  for (i in 1:num.classes) {
    confusion.matrix[i, num.classes + 1] <-
      confusion.matrix[i, i] / sum(confusion.matrix[i, 1:num.classes])
    confusion.matrix[num.classes + 1, i] <-
      confusion.matrix[i, i] / sum(confusion.matrix[1:num.classes, i])
  }
  return(confusion.matrix)
}


#
# Random Discriminant
#
require(ppls)

fit.gaussian <- function(x, n) {
  # Args:
  #   x: a list or 1-dimensional vector
  #   n: the total number of observations in the set the members of x come from

  gaussian.model <- list(
    mu = mean(x),
    sigma = var(x),
    prior = length(x) / n
  )

  if (gaussian.model$prior == 0) {
    gaussian.model$mu <- 1
    gaussian.model$sigma <- 1
  } else if (is.na(gaussian.model$sigma)) {
    gaussian.model$sigma <- 1  # arbitrary, hack for when fittint to 1 element
  }

  return(gaussian.model)
}

relative.gaussian.pdf <- function(model, x) {
  # Args:
  #   model:
  #   x:

  scaling.factor <- 1 / (sqrt(2 * pi) * model$sigma)
  exponent <- -(((x - model$mu) ^ 2) / (2 * model$sigma ^ 2))
  probability <- model$prior * scaling.factor * exp(exponent)

if (is.na(probability)) {
print(model)
print(x)
}
  return(probability)
}

construct.random.discriminant <- function(classification.frame) {
  # Args:
  #   data: labeled data in rows

  dimensionality <- ncol(classification.frame) - 2

  projection.line <- normalize.vector(
    matrix(runif(n = dimensionality, min = -5, max = 5), ncol = 1)
  )

  data.c1 <- data.matrix(subset(classification.frame, target == 1)[, -(1:2)])
  data.c2 <- data.matrix(subset(classification.frame, target == 2)[, -(1:2)])

  projections.c1 <- data.c1 %*% projection.line
  projections.c2 <- data.c2 %*% projection.line

  gaussian.1 <- fit.gaussian(x = projections.c1, n = nrow(classification.frame))
  gaussian.2 <- fit.gaussian(x = projections.c2, n = nrow(classification.frame))

  return(list(
    projection.line = projection.line,
    decision.boundary = NULL,#0.5 * (mu.0 + mu.1),  # TODO: make this better!!!!!!!!!!!!!!!!!!!!!
    class.1.model = gaussian.1,
    class.2.model = gaussian.2
  ))
}

decision.boundary.classify <- function(x, decision.boundary) {
  # Args:
  #   x:
  #   decision.boundary:

  if (x >= decision.boundary) {
    return(1)
  } else {
    return(2)
  }
}

discriminant.predict <- function(model, x) {
  # Args:
  #   model: an object that represents an LDA model
  #   x: the data to be classified via LDA

  if (is.null(model$decision.boundary)) {
    predictions <- list()
    for (i in 1:nrow(x)) {

print(dim(x[i, ]))
print(dim(model$projection.line))
      proj <- data.matrix(x[i, ]) %*% model$projection.line

      p.class.1 <-
        relative.gaussian.pdf(model = model$class.1.model, x = proj)
      p.class.2 <-
        relative.gaussian.pdf(model = model$class.2.model, x = proj)

      if (p.class.1 >= p.class.2) {
        predictions[[i]] <- 1
      } else {
        predictions[[i]] <- 2
      }
    }
    return(predictions)
  } else {
    projections <- data.matrix(x) %*% model$projection.line
    return(
      lapply(projections, decision.boundary.classify, model$decision.boundary)
    )
  }
}


#
# Random Tree
#
construct.stopping.criterion <- function(which = 'depth', threshold = 0.99) {
  # Args:
  #   which: determines which criterion to use
  #   threshold: indicates at what threshold tree building should stop; for
  #              measures like entropy, gini index, and classification purity,
  #              a number (0, 1) is sufficient, for a measure like depth
  #              integers should be used.

  specific.criterion <- NULL
  if (which == 'depth') {
    specific.criterion <- function(classification.frame, class = 0, depth = -1) {
      return(depth > threshold)
    }
  } else if (which == 'entropy') {
    specific.criterion <- function(classification.frame, class = 0, depth = -1) {
      n <- nrow(classification.frame)
      n.1 <- nrow(subset(classification.frame)) / n
      n.2 <- n - n.1

      entropy <- -((n.1 * log2(n.1)) + (n.2 * log2(n.2)))
print(entropy)
      return(entropy <= threshold)
    }
  } else {  #if (which == 'prediction')
    specific.criterion <- function(classification.frame, class = 0, depth = -1) {
      purity <- nrow(subset(classification.frame, target == class)) /
        nrow(classification.frame)

      purity <- max(purity, 1 - purity)

      return(purity >= threshold)
    }
  }

  f <- function(classification.frame = NULL, class = 0, depth = -1) {
    if (nrow(classification.frame) <= 1) {
      return(T)
    } else if (all(classification.frame$target == 1) ||
               all(classification.frame$target == 2)) {
      return(T)
    } else {
      return(specific.criterion(classification.frame, class, depth))
    }
  }

  return(f)
}

construct.random.subtree <-
  function(classification.frame, stopping.criterion, depth = 1) {
  # Args:
  #   data: a frame of labeled data
  #   stopping.criterion: a function that will determine if the tree should
  #                       continue building. Could be entropy, gini index, etc.

  if (stopping.criterion(classification.frame, class = 1, depth = depth)){
    return(NULL)
  }

  discriminant.model <- construct.random.discriminant(classification.frame)

print(classification.frame)

  classification.frame$prediction <- discriminant.predict(
    model = discriminant.model,
    x = classification.frame[, -(1:2)]
  )

print(classification.frame)

  predicted.as.1 <- subset(classification.frame, prediction == 1)
  predicted.as.1$prediction <- rep(0, nrow(predicted.as.1))
  class.1.branch <- construct.random.subtree(
    classification.frame = predicted.as.1,
    stopping.criterion,
    depth = depth + 1
  )

  predicted.as.2 <- subset(classification.frame, prediction == 2)
  predicted.as.2$prediction <- rep(0, nrow(predicted.as.2))
  class.2.branch <- construct.random.subtree(
    classification.frame = predicted.as.2,
    stopping.criterion,
    depth = depth + 1
  )

  subtree <- list(
    model = model,
    split = length(predicted.as.1) / nrow(classification.frame),
    class.1.branch = class.1.branch,
    class.2.branch = class.2.branch
  )

  return(subtree)
}

compute.entropy <- function(c.frame) {

  n <- nrow(c.frame)

  if (n == 0) {
    return(0)
  }

  n.1 <- nrow(subset(c.frame, target == 1))
  n.2 <- n - n.1

  if (n.1 == 0 || n.2 == 0) {
    return(0)
  }

  p.1 <- n.1 / n
  p.2 <- n.2 / n

  entropy <- -((p.1 * log2(p.1)) + (p.2 * log2(p.2)))
  return(entropy)
}

construct.random.tree <- function(c.frame) {
  # Args:
  #   data: a frame of labeled data
  #   stopping.criterion: a function that will determine if the tree should
  #                       continue building. Could be entropy, gini index, etc.

  entropy <- compute.entropy(c.frame)
print(entropy)

  if (nrow(c.frame) <= 1 || entropy <= 0.2){
    return(NULL)
  }

  discriminant.model <- construct.random.discriminant(c.frame)

  c.frame$prediction <- discriminant.predict(
    model = discriminant.model,
    x = c.frame[, -(1:2)]
  )

  predicted.as.1 <- subset(c.frame, prediction == 1)
  predicted.as.1$prediction <- rep(0, nrow(predicted.as.1))
  class.1.branch <- construct.random.tree(c.frame = predicted.as.1)

  predicted.as.2 <- subset(c.frame, prediction == 2)
  predicted.as.2$prediction <- rep(0, nrow(predicted.as.2))
  class.2.branch <- construct.random.tree(c.frame = predicted.as.2)

  tree <- list(
    model = model,
    split = nrow(predicted.as.1) / nrow(c.frame),
    class.1.branch = class.1.branch,
    class.2.branch = class.2.branch
  )

  return(tree)
}

random.tree.predict <- function(random.tree, classification.frame) {
  # Args:
  #
  #
  #

  if (!is.null(random.tree) && (nrow(classification.frame) > 0)) {
    classification.frame$prediction <- discriminant.predict(
      model = random.tree$model,
      x = classification.frame[, -(1:2)]
    )

    predicted.as.1 <- subset(classification.frame, prediction == 1)
    predicted.as.2 <- subset(classification.frame, prediction == 2)

    from.branch.1 <-
      random.tree.predict(random.tree$class.1.branch, predicted.as.1)
    from.branch.2 <-
      random.tree.predict(random.tree$class.2.branch, predicted.as.2)

    predictions <- rbind(
      from.branch.1,
      from.branch.2
    )

    return(predictions)
  } else {
    return(classification.frame)
  }
}


#
# Main
#

data <- get.data(
  train.file = ZIP.TRAIN.FILE.NAME,
  test.file = ZIP.TEST.FILE.NAME,
  class.1 = 1,
  class.2 = 2
)




tree.1.2 <- construct.random.tree(c.frame = data$train)

write.table(x = data$train, file = '1_2_pca.csv', sep = ',')











#
# 'Unit Tests'
#
d <- matrix(
  rbind(c(1, 0.5), c(1, 0.6), c(1, 0.4), c(-1, 0.5), c(-1, 0.4), c(-1, 0.6)),
  ncol = 2
)
f <- construct.classification.frame(data = d, targets = c(1, 1, 1, 2, 2, 2))
model <- construct.random.discriminant(f)
easy.model <- model
easy.model$projection.line <- matrix(c(1,0), ncol = 1)
easy.model$decision.boundary <- 0
discriminant.predict(model, f[, -(1:2)])
discriminant.predict(easy.model, f[, -(1:2)])


# subtree <- construct.random.subtree(
#   classification.frame = f,
#   stopping.criterion =
#     construct.stopping.criterion(which = 'entropy', threshold = 0.3)
# )

tree <- construct.random.tree(
  c.frame = f
)

random.tree.predict(random.tree = subtree, f)
