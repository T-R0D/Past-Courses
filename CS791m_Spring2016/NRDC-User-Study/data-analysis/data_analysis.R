##
#
##

library(ggplot2)
library(reshape2)

# print(getSrcDirectory(main))  # doesn't seem to work so we define WD manually
WD <- 'C:/Users/Terence/Documents/GitHub/NRDC-User-Study/data-analysis/'

FULL.DATA.FILE <- paste(WD, '../data/full_data.csv', sep='')


get.data.file <- function(full.path.file.name) {
  # Places the input file in a data.frame, conveniently formatted.
  #
  # Args:
  #   full.path.file.name: The name of the file with fully qualified path.
  #
  # Returns:
  #   A data frame containing all of the file's data in a format ready for
  #   analysis.

  data <- read.csv(full.path.file.name, header=T, sep =',')

  version.relable <- function(x) {
    if (x == 'A') {
      return(0)
    } else {
      return(1)
    }
  }

  yes.no.relable <- function(x) {
    if (x == 'Y' || x == 'y') {
      return(1)
    } else {
      return(0)
    }
  }
  give.up.complete.relable <- function(x) {
    if (x == 'Y' || x == 'y') {
      return(1)
    } else {
      return(0)
    }
  }

  data$version <- lapply(data$version, version.relable)
  data$used_geospatial_tutorial <- lapply(data$used_geospatial_tutorial, yes.no.relable)
  data$used_geospatial_site_list <- lapply(data$used_geospatial_site_list, yes.no.relable)

  # TODO(Terence): convert the task times from milliseconds to seconds?

  return(data)
}


get.task.set.data <- function(full.data, task.set=0, interface=0, which='time') {
  # Extracts the data for one "version" of a task from the full data.
  #
  # Args:
  #   full.data: A data.frame with all of the data
  #   task.set: which task set, 1 or 2
  #   interface: the interface version used, A or 1
  #   which: which data to get, time, outcome, or clicks
  #
  # Return:
  #   A data.frame containing the appropriate subset of data.

  if (task.set != 0 && task.set != 1) {
    stop('task.set must be 1 or 2')
  }

  data <- subset(full.data, version==interface)

  if (task.set == 0) {
    if (which == 'time') {
      data <- subset(data, select=t_00_00:t_00_06)
    } else if (which == 'outcome') {
      data <- subset(data, select=p_00_00:p_00_06)
    } else {
      data <- subset(data, select=c_00_00:c_00_06)
    }
  } else {
    if (which == 'time') {
      data <- subset(data, select=t_01_00:t_01_06)
    } else if (which == 'outcome') {
      data <- subset(data, select=p_01_00:p_01_06)
    } else {
      data <- subset(data, select=c_01_00:c_01_06)
    }
  }

  data$round <- rep(if (task.set == 0) {'first'} else {'second'}, nrow(data))
  data$version <- rep(if (interface == 0) {'current'} else {'in-development'}, nrow(data))

  return(data)
}


get.task.for.interface <- function(full.data, version) {
  COL.NAMES = c('T1','T2','T3','T4','T5','T6','T7', 'round', 'version')

  #
  # TIMES
  #
  current.first <- get.task.set.data(full.data, task.set=0, interface=0, which='time')
  dev.first <- get.task.set.data(full.data, task.set=0, interface=1, which='time')
  current.second <- get.task.set.data(full.data, task.set=1, interface=0, which='time')
  dev.second <- get.task.set.data(full.data, task.set=1, interface=1, which='time')

  colnames(current.first) <- COL.NAMES
  colnames(dev.first) <- COL.NAMES
  colnames(current.second) <- COL.NAMES
  colnames(dev.second) <- COL.NAMES


  # ALL
  task.times <- rbind(current.first, dev.first, current.second, dev.second)

  print(colMeans(subset(task.times, select=T1:T7, version=='current')))
  print(colMeans(subset(task.times, select=T1:T7, version=='in-development')))

  X <- task.times$version
  Y <- cbind(task.times$T1, task.times$T2, task.times$T3, task.times$T4, task.times$T5, task.times$T6, task.times$T7)
  model <- aov(Y ~ X, task.times)
  summary(model)

  task.times <- melt(task.times)
  task.times$value <- task.times$value / 1000

  plot <- ggplot(task.times, aes(x=variable, y=value))
  plot <- plot + ggtitle('Overall Task Completion Time')
  plot <- plot + labs(x='Task', y='Time to Complete (s)')
  plot <- plot + geom_boxplot(aes(color=factor(version)), outlier.shape=NA)
  plot <- plot + scale_y_continuous(limits=c(0, 350))
  ggsave(paste(WD, 'overall_task_completion.png', sep=''), plot=plot)

  # INITIAL
  task.times <- rbind(current.first, dev.first)

  X <- task.times$version
  Y <- cbind(task.times$T1, task.times$T2, task.times$T3, task.times$T4, task.times$T5, task.times$T6, task.times$T7)
  model <- aov(Y ~ X, task.times)
  summary(model)

  task.times <- melt(task.times)
  task.times$value <- task.times$value / 1000

  plot <- ggplot(task.times, aes(x=variable, y=value))
  plot <- plot + ggtitle('First Round Task Completion Time')
  plot <- plot + labs(x='Task', y='Time to Complete (s)')
  plot <- plot + geom_boxplot(aes(color=factor(version)), outlier.shape=NA)
  plot <- plot + scale_y_continuous(limits=c(0, 350))
  ggsave(paste(WD, 'first_task_completion.png', sep=''), plot=plot)


  # SECOND
  task.times <- rbind(current.first, dev.first, current.second, dev.second)

  X <- task.times$version
  Y <- cbind(task.times$T1, task.times$T2, task.times$T3, task.times$T4, task.times$T5, task.times$T6, task.times$T7)
  model <- aov(Y ~ X, task.times)
  summary(model)

  task.times <- melt(task.times)
  task.times$value <- task.times$value / 1000

  plot <- ggplot(task.times, aes(x=variable, y=value))
  plot <- plot + ggtitle('Second Round Task Completion Time')
  plot <- plot + labs(x='Task', y='Time to Complete (s)')
  plot <- plot + geom_boxplot(aes(color=factor(version)), outlier.shape=NA)
  plot <- plot + scale_y_continuous(limits=c(0, 250))
  ggsave(paste(WD, 'second_task_completion.png', sep=''), plot=plot)


  #
  # CLICKS
  #
  current.first <- get.task.set.data(full.data, task.set=0, interface=0, which='clicks')
  dev.first <- get.task.set.data(full.data, task.set=0, interface=1, which='clicks')
  current.second <- get.task.set.data(full.data, task.set=1, interface=0, which='clicks')
  dev.second <- get.task.set.data(full.data, task.set=1, interface=1, which='clicks')

  colnames(current.first) <- COL.NAMES
  colnames(dev.first) <- COL.NAMES
  colnames(current.second) <- COL.NAMES
  colnames(dev.second) <- COL.NAMES


  # ALL
  task.times <- rbind(current.first, dev.first, current.second, dev.second)

  print(colMeans(subset(task.times, select=T1:T7, version=='current')))
  print(colMeans(subset(task.times, select=T1:T7, version=='in-development')))

  X <- task.times$version
  Y <- cbind(task.times$T1, task.times$T2, task.times$T3, task.times$T4, task.times$T5, task.times$T6, task.times$T7)
  model <- aov(Y ~ X, task.times)
  summary(model)

  task.times <- melt(task.times)

  plot <- ggplot(task.times, aes(x=variable, y=value))
  plot <- plot + ggtitle('Overall Task Clicks')
  plot <- plot + labs(x='Task', y='Clicks to Complete')
  plot <- plot + geom_boxplot(aes(color=factor(version)), outlier.shape=NA)
  plot <- plot + scale_y_continuous(limits=c(0, 150))
  ggsave(paste(WD, 'overall_task_clicks.png', sep=''), plot=plot)


  # INITIAL
  task.times <- rbind(current.first, dev.first)

  X <- task.times$version
  Y <- cbind(task.times$T1, task.times$T2, task.times$T3, task.times$T4, task.times$T5, task.times$T6, task.times$T7)
  model <- aov(Y ~ X, task.times)
  summary(model)

  task.times <- melt(task.times)

  plot <- ggplot(task.times, aes(x=variable, y=value))
  plot <- plot + ggtitle('First Round Task Clicks')
  plot <- plot + labs(x='Task', y='Clicks to Complete')
  plot <- plot + geom_boxplot(aes(color=factor(version)), outlier.shape=NA)
  # plot <- plot + scale_y_continuous(limits=c(0, 140))
  ggsave(paste(WD, 'first_task_clicks.png', sep=''), plot=plot)


  # SECOND
  task.times <- rbind(current.first, dev.first, current.second, dev.second)

  X <- task.times$version
  Y <- cbind(task.times$T1, task.times$T2, task.times$T3, task.times$T4, task.times$T5, task.times$T6, task.times$T7)
  model <- aov(Y ~ X, task.times)
  summary(model)

  task.times <- melt(task.times)

  plot <- ggplot(task.times, aes(x=variable, y=value))
  plot <- plot + ggtitle('Second Round Task Clicks')
  plot <- plot + labs(x='Task', y='Clicks to Complete')
  plot <- plot + geom_boxplot(aes(color=factor(version)), outlier.shape=NA)
  # plot <- plot + scale_y_continuous(limits=c(0, 75))
  ggsave(paste(WD, 'second_task_clicks.png', sep=''), plot=plot)
}


analyze.task.times <- function(full.data) {
  # Perform the analysis for the task timings.
  #
  #



}

get.questionnaire.comparison.data <- function(full.data) {
  # Constructs a data frame of the data derived from the comparison questions
  #
  # Args:
  #   full.data: the full data data.frame.
  #
  # Returns:
  #   A data frame that indicates a score for each of the comparison questions.

  data <- subset(full.data, select=Q1.comp:Q4.comp)

  rating.complement <- function(x, max.rating=5, min.rating=1) {
    return(max.rating - x + min.rating)
  }

  orig.rows <- nrow(data)
  orig.cols <- ncol(data)
  for (i in 1:orig.rows) {
    orig <- data[i, ]
    new <- list()
    for (j in 1:orig.cols) {
      new[[j]] <- rating.complement(orig[[j]])
    }
    data <- rbind(data, new)
  }

  data$version <- factor(c(rep(1, orig.rows), rep(0, orig.rows)),
                         labels=c('Current', 'In-Development'))

  colnames(data) <- c('Q1', 'Q2', 'Q3', 'Q4', 'version')

  return(data)
}


analyze.comparison.data <- function(full.data) {
  # Perform analysis of the direct subjective comparisons.
  #
  # Args:
  #   full.data:

  comp.data <- get.questionnaire.comparison.data(full.data)

  # PLOTTING ###
  c.mean.data <- colMeans(data.matrix(subset(comp.data, version==1)))

  # c.data <- melt(comp.data)
  # plot <- ggplot(c.data, aes(version, value))

  plot <- ggplot(subset(c.data, version==1), aes(x=variable, y=value))
  plot <- plot + ggtitle('Direct Comparison of Subjective Usability') +
   labs(x='Survey Question', y='Preference')

  plot <- plot + geom_violin(draw_quantiles = c(0.25, 0.5, 0.75), fill = "grey80", colour = "#3366FF")

  ggsave(paste(WD, 'direct_comparison_violin.png',sep=''), plot=plot)

  # ANOVA ###
  Y <- cbind(comp.data$Q1, comp.data$Q2, comp.data$Q3, comp.data$Q4)
  X <- comp.data$version

  model <- aov(Y ~ X)

  print('\nQUESTIONNAIRE DIRECT COMPARISON ANOVA\n')
  summary(model)

}












data <- get.data.file(FULL.DATA.FILE)

main <- function() {
  # The main function.

  data <- get.data.file(FULL.DATA.FILE)

  for (version in c('A', 1)) {
    print('==============')
    for (t in c(1,2)) {
      for (v in c('time', 'clicks')) {
        print(colMeans(get.task.set.data(data, task.set=t, interface=version,which=v)))
      }
    }
  }


}

main()
