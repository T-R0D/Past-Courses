##################
# STAT775
# HW 03
# Exercise 02
#
# Perform Ridge Regression
# and explore the performance
# of using different
# lambda values. Plot the
# results.
#
##################

#
# Initial Setup
#
setwd("C:/Users/Terence/Documents/GitHub/STAT775/HW03")
# setwd("~/STAT775/HW03/")

RESULTS.FILE.NAME = "HW03_Exercise02.png"
DATA.FILE.NAME <- "../DataSets/prostate_cancer/prostate_cancer_data"
# no intercept will be applied since we are performing
# ridge regression
FEATURE.INDICES <- 1:8
TARGET.INDEX <- 9
TRAIN.SET.FLAG <- 10


data <- read.table(DATA.FILE.NAME)
# according to the dataset info, this needs to be done
data[, FEATURE.INDICES] <- scale(data[, FEATURE.INDICES], center = T, scale = T)

# store data with rows as observations
train.data <- data.matrix(subset(data, data$train == T))[, FEATURE.INDICES]
intercepts <- rep(1, nrow(train.data))
train.data.with.intercepts <- cbind(intercepts, train.data)
train.targets <- data.matrix(subset(data, data$train == T))[, TARGET.INDEX]

test.data <- data.matrix(subset(data, data$train == F))[, FEATURE.INDICES]
intercepts <- rep(1, nrow(test.data))
test.data.with.intercepts <- cbind(intercepts, test.data)
test.targets <- data.matrix(subset(data, data$train == F))[, TARGET.INDEX]


#
# Ridge Regression
#
ridge.regression <- function(x, y, lambda) {
 #
 # Args:
 #   x: a matrix of training observations where observations are rows
 #   y: a column vector of regression targets
 #   lamba: the ridge coefficient; larger values reduce the effect of less
 #          relevant features

  lambda.I <- diag(lambda, nrow = ncol(x))
  model.betas <- solve( (t(x) %*% x) + lambda.I) %*% t(x) %*% y
  model.betas <- rbind(mean(y), model.betas)

  return(list(
    beta.hat = matrix(model.betas, nrow = 1)
  ))
}

predict <- function(model, x) {
  #
  # Args:
  #   model: an n x 1 column vector of regression coefficients
  #   x: a ? x n matrix of observation data

  return(model$beta.hat %*% t(x))
}

#
# Main
#
lambda.values <- seq(from = 0, to = 8, by = .5)
ridge.regression.results <- data.frame(
  row.names = lapply(as.list(lambda.values), toString),
  'Lambda' = lambda.values,
  'RSS' = rep(0, length(lambda.values)),
  stringsAsFactors = F
)

trial <- 1
for (lambda in lambda.values) {
   model <- ridge.regression(x = train.data, y = train.targets, lambda = lambda)
#   model <- list(beta.hat = matrix(c(2.452, 0.42, 0.238, -0.046, 0.162, 0.227, 0.000, 0.040, 0.133), nrow = 1))
  predictions <- predict(model, cbind(1, train.data))

  ridge.regression.results[toString(lambda), 'RSS'] <-
    sum((matrix(train.targets, nrow = 1) - predictions) ^ 2)
}







solution <- c(2.452, 0.42, 0.238, -0.046, 0.162, 0.227, 0.000, 0.040, 0.133)
print(solution)
print(round(
  ridge.regression(
    x = train.data[, (1:8) + 1],
    y = train.targets,
    lambda = 5
  )$beta.hat,
  3
))
print("--------")
print(
  ridge.regression(
    x = train.data,
    y = train.targets,
    lambda = 5
  )$beta.hat -
  solution
)
# 2.452
# 0.420
# 0.238
# -0.046
# 0.162
# 0.227
# 0.000
# 0.040
# 0.133

#
# Test errors
#
lambda.sequence <- seq(-5, 50, 0.5)
lambdas <- rep(0, length(lambda.sequence))
errors <- rep(0, length(lambda.sequence))

ridge.regression.results <- data.frame(
  Lambda = lambdas,
  Error = errors,
  stringsAsFactors = F
)

index <- 1
for (lambda in lambda.sequence) {
  betas <- fit.ridge.regression.model(
    train.dat = train.data,
    targets = train.targets,
    l = lambda
  )

  residual.sum.of.squares <- 0.0
  for (i in 1:nrow(train.data)) {
    prediction <- t(betas) %*% data.matrix(train.data.with.intercepts[i, ])
    residual.sum.of.squares <-
      residual.sum.of.squares + (train.targets[i] - prediction)^2
  }

  ridge.regression.results$Lambda[[index]] <- lambda
  ridge.regression.results$Error[[index]] <- residual.sum.of.squares
  index = index + 1
}


#
# Plot Results
#
library(ggplot2)

plot.theme <- theme(
  plot.background = element_blank(),
  panel.grid.major = element_blank(),
  panel.grid.minor = element_blank(),
  panel.border = element_blank(),
  panel.background = element_blank(),
  axis.line = element_line(size=.4),
  axis.title.x = element_text(face="bold", color="black", size=10),
  axis.title.y = element_text(face="bold", color="black", size=10),
  plot.title = element_text(face="bold", color = "black", size=12)
)

error.plot <- ggplot(
  ridge.regression.results,
  aes(x = Lambda, y = Error)
) +
  plot.theme +
  geom_point(shape = 16, size = 3) +
  geom_line() +
  scale_x_continuous(
    limits = c(-5, 50),
    breaks = seq(-5, 50, 5)
  ) +
  scale_y_continuous(
    limits = c(35, 80),
    breaks = seq(35, 80, 5)
  ) +
  labs(
    title = "Errors for Ridge Regressions Using Varying Lambda Values",
    x = "Lambda",
    y = "Residual Sum-of-Squares"
  )
error.plot

ggsave(filename = RESULTS.FILE.NAME, plot = error.plot)