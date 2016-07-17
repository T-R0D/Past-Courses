##################
# STAT775
# HW 03
# Exercise 01
#
# Reproduce Figure 3-5 from the
# Elements of Statistical Learning
# (This is a scatterplot of the
# errors produced by models built
# from all possible subsets of
# predictors, except for an
# intercept, for simplicity)
#
##################

#
# Initial Setup
#
setwd("C:/Users/Terence/Documents/GitHub/STAT775/HW03/")
# setwd("~/STAT775/HW03/")

RESULTS.FILE.NAME = "HW03_Exercise01.png"
DATA.FILE.NAME <- "../DataSets/prostate_cancer/prostate_cancer_data"
# these include the intercept that gets added to the matrices
FEATURE.INDICES <- 1:9
TARGET.INDEX <- 10
TRAIN.SET.FLAG <- 11


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
    beta.hat = solve(t(x) %*% x) %*% t(x) %*% y
  ))
}

predict <- function(model, x) {
 #
 # Args:
 #   model: an n x 1 column vector of regression coefficients
 #   x: a ? x n matrix of observation data

  return(t(model$beta.hat) %*% t(x))
}

#
# Main
#
require('sets')
feature.subsets <- set_power(as.set(2:9))  # set of features excluding intercept

# setup a results data.frame - no results yet, just placeholders
k <- rep(0, length(feature.subsets))
subset.as.string <- rep("", length(feature.subsets))
error <- rep(0, length(feature.subsets))
best.for.k <- rep(F, length(feature.subsets))

subset.selection.results <- data.frame(
  K = k,
  Subset = subset.as.string,
  Error = error,
  Best.For.K = best.for.k,
  stringsAsFactors = F
)

# perform regression for all subsets of features

# valid <- rbind(train.data, test.data)
# v.t <- rbind(matrix(train.targets, ncol = 1) , matrix(test.targets, ncol = 1))

index <- 1
for (subset in feature.subsets) {
  features.to.use <- c(1, subset)  # column 1 for intercept term

  model <- ols.regression(
    x = train.data[, unlist(features.to.use)],
    y = train.targets
  )
  predictions <- predict(model, train.data[, unlist(features.to.use)])

  RSS <- sum((matrix(train.targets, nrow = 1) - predictions) ^ 2)

  subset.selection.results$K[[index]] <- length(subset)
  subset.selection.results$Subset[[index]] <-
    if(set_is_empty(subset)){"Empty"} else{subset}
  subset.selection.results$Error[[index]] <- RSS

  index = index + 1
}

# tag best subset for each size in the results data.frame
for (i in 0:length(0:8)) {
  results.for.subset.size <- subset(subset.selection.results, K == i)
  min.error.rowname <- row.names(
    results.for.subset.size[which.min(results.for.subset.size$Error), ]
  )
  subset.selection.results[min.error.rowname, 'Best.For.K'] <- T
}


# plot results
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

subset.error.plot <- ggplot(
  subset.selection.results,
  aes(x = K, y = Error, group = Best.For.K)
) +
  plot.theme +
  geom_point(shape = 16, size = 3) +
  geom_point(aes(color = Best.For.K)) +
  geom_line(aes(alpha = Best.For.K, color = Best.For.K)) +
  scale_color_manual(
    name = "Selection",
    labels = c("Other Subsets", "Best Subset"),
    values = c("green", "red")
  ) +
  scale_shape_manual(
    name = "Selection",
    labels = c("Other Subsets", "Best Subset"),
    values = c(16, 8)
  ) +
  scale_alpha_discrete(guide = F) + #continuous
  scale_y_continuous(
    limits = c(0, 100),
    breaks = seq(0, 100, 10)
  ) +
  labs(
    title = "Errors for Subsets of Predictors",
    x = "Subset Size k",
    y = "Residual Sum-of-Squares"
  )

ggsave(filename = RESULTS.FILE.NAME, plot = subset.error.plot)