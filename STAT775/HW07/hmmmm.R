DATA.PATH <- "../DataSets/zip.data/"
ZIP.TRAIN.FILE.NAME <- paste0(DATA.PATH, "zip.train")
ZIP.TEST.FILE.NAME <- paste0(DATA.PATH, "zip.test")

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

  test[, -1] <- predict(pca.model, test[, -1])
  test <- test[, 1:(num.principle.components + 1)]

  return(list(
    train = train,
    test = test
  ))
}

data <- get.data(
  train.file = ZIP.TRAIN.FILE.NAME,
  test.file = ZIP.TEST.FILE.NAME,
  class.1 = 1,
  class.2 = 2
)



library('party')

start <- proc.time()
print(proc.time() - start)

NUM.TREES <- 21
trees <- list()
for (i in 1:NUM.TREES) {
  trees[[i]] <- ctree(V1 ~ ., data = data$train)
}




tree.1.2 <- ctree(V1 ~ ., data = train)
plot(tree.1.2)
print(tree.1.2)

predictions <- round(predict(tree.1.2))



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

print(compute.confusion.matrix(predictions, train[, 1]))
