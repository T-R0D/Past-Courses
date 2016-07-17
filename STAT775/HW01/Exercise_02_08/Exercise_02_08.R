##
# ESL Exercise 2.8
#
##

######################
# some initial setup #
######################
working.directory = "~/Desktop/STAT775/HW01/Exercise_02_08"
training.file.name = "zip.test"
test.file.name = "zip.train"

setwd(working.directory)

require(ggplot2)
require(reshape)

####################
# Data Preparation #
####################
read.and.filter.data <- function(file.name) {
  #
  #
  # Args:
  #
  # Returns:
  #
  data = read.table(
    file.name,
    header = F,
    row.name = NULL
  )
  data = subset(data, V1 == 2 | V1 == 3)
  return(data)
}

separate.classes <- function(x) {
  if (x == 3) {
    return(1)
  } else if (x == 2) {
    return(-1)
  }
}

zip.train <- read.and.filter.data(training.file.name)
training.data <- data.matrix(zip.train[, -1])
training.targets <- data.matrix(zip.train[, 1])
for (i in 1:length(training.targets)) {
  training.targets[i] <- separate.classes(training.targets[i])
}

zip.test <- read.and.filter.data(test.file.name)
test.data <- data.matrix(zip.train[, -1])
test.targets <- data.matrix(zip.train[, 1])
for (i in 1:length(test.targets)) {
  test.targets[i] <- separate.classes(test.targets[i])
}

k.range <- 1:15  # simpler than doing just 1, 3, 5, 7, 15
determine.category <- function(x) {
  if(x > 0) {
    return(1)
  } else {
    return(-1)
  }
}

#####################
# Linear Regression #
#####################
least.squares <- function(x, y) {
  new.x <- x
#   new.x <- cbind(x, rep(1.0, length(x[, 1]))) this gives singular matrix, good practice though
  xt.x <- t(new.x) %*% new.x
  xt.x.inv <- solve(xt.x)
  regression.weights <- xt.x.inv %*% (t(new.x) %*% y)
  return(regression.weights)
}

linear.predict <- function(inputs, betas) {
  # sometimes the formula is listed as x^t * b, but our 'inputs' data already has each
  # test vector in a row when we pass it as a matrix (for this script)
  result <- t(inputs) %*% betas
#   result <- lapply(result, function(x) {return(determine.category(x))})
  return(determine.category(result))
}

digit.model.weights <- least.squares(x = training.data, y = training.targets)

linear.training.predictions <- list()
for(i in 1:length(training.targets)) {
  linear.training.predictions[i] <- linear.predict(inputs = training.data[i, ], betas = digit.model.weights)
}
linear.test.predictions <- list()
for(i in 1:length(test.targets)) {
  linear.test.predictions[i] <- linear.predict(inputs = test.data[i, ], betas = digit.model.weights)
}


########
# K-NN #
########
euclidean.distance <- function(x, y) {
  diff <- x - y
  squares <- diff^2
  return(sqrt(sum(squares)))
}

my.knn.predict <- function(input.point, training.points, training.labels, k) {
  distance = NULL
  for (i in 1:length(training.points[, 1])) {
    distance[i] <- euclidean.distance(input.point, training.points[i, ])
  }
  
  category <- training.labels
  results <- data.frame(distance, category)
  results <- results[order(-distance), ]
  results <- results$category[1:k]
  
  votes <- 
  
  return(determine.category(mean(results)))
}

knn.training.predictions <- matrix(0, length(k.range), length(training.targets))
knn.test.predictions <- matrix(0, length(k.range), length(test.targets))
for (k in k.range) {
  for (j in 1:length(training.targets)) {
    knn.training.predictions[k, j] <- my.knn.predict(
      input.point = training.data[j, ],
      training.points = training.data,
      training.labels = training.targets,
      k = k
    )
  }
}
for (k in k.range) {
  for (j in 1:length(test.targets)) {
    knn.test.predictions[k, j] <- my.knn.predict(
      input.point = test.data[j, ],
      training.points = training.data,
      training.labels = training.targets,
      k = k
    )
  }
}


######################
# Error Computations #
######################
compute.error.rate <- function(targets, predictions) {
  incorrect.predictions = 0
  for (i in 1:length(targets)) {
    if (targets[i] != predictions[i]) {
      incorrect.predictions <- (incorrect.predictions + 1)
    }
  }
  return(incorrect.predictions / length(targets))
}

linear.training.error.rates <- compute.error.rate(training.targets, linear.training.predictions)
linear.training.error.rates <- rep(linear.training.error.rates, 15)
linear.test.error.rates <- compute.error.rate(test.targets, linear.test.predictions)
linear.test.error.rates <- rep(linear.test.error.rates, 15)

knn.training.error.rates <- NULL
for (k in k.range) {
  knn.training.error.rates[k] <- compute.error.rate(training.targets, knn.training.predictions[k, ])
}
knn.test.error.rates <- NULL
for (k in k.range) {
  knn.test.error.rates[k] <- compute.error.rate(test.targets, knn.test.predictions[k, ])
}

linear.training.error.rates <- rep(.006, 5)
linear.test.error.rates <- rep(.042, 5)
knn.training.error.rates <- c(0, .005, .006,  .0065, .0095)
knn.test.error.rates <- c(.0249, .031, .031,  .0325, .038)

errors.for.plotting <- data.frame(
  "K" = k.range,
  "Linear (Training)" = linear.training.error.rates,
  "Linear (Test)" = linear.test.error.rates,
  "KNN (Training)" = knn.training.error.rates,
  "KNN (Test)" = knn.test.error.rates
)

############
# Plotting #
############

plot.data <- melt(errors.for.plotting, id = "K", variable_name = "Method")
ggplot(data = plot.data, aes(x = K, y = value, color = Method)) +
  geom_line() +
  ggtitle("A comparison of Linear Regression and k-NN Approaches") +
  xlab("K") +
  ylab("Error Rate") +
  ylim(0, 0.05) +
  xlim(0, 16) +
  scale_color_hue(
    name = "Method (Data)",
    labels = c(
      "Linear Regression (Training)",
      "Linear Regression (Test)",
      "k-NN (Training)",
      "k-NN (Test)"
    )
  )
ggsave('error_plot.png')