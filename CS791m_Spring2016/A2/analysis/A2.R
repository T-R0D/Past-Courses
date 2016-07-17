##
#
##

library(ggplot2)
library(reshape2)

WD <- 'C:/Users/Terence/Documents/GitHub/CS791m_Spring2016/A2/'
LETTER.GUESS.FILE <- paste(WD, 'data/_letter-guess.txt', sep='')
REACTION.FILE <- paste(WD, 'data/_reaction.txt', sep='')
MATCHING.FILE <- paste(WD, 'data/_matching.txt', sep='')


multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  print('This multiplot function does not currently support saving the multiplot to a file. Please save the files manually.')

  ##
  # Code from
  # http://www.cookbook-r.com/Graphs/Multiple_graphs_on_one_page_(ggplot2)/
  ##
  library(grid)

  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)

  numPlots = length(plots)

  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }

  if (numPlots==1) {
    print(plots[[1]])

  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))

    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))

      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}


###
# 2-3
###

get.reduced.words <- function(clean.text, reduced.text) {
  ##
  #
  ##
  word.boundaries <- list()
  chars <- unlist(strsplit(clean.text, '*'))
  for (i in 1:length(chars)) {
    if (chars[i] == ' ') {
      word.boundaries <- c(word.boundaries, i)
    }
  }

  boundary.marker <- '!'
  reduced.chars <- unlist(strsplit(reduced.text, '*'))
  for (i in word.boundaries) {
    reduced.chars[i] <- boundary.marker
  }

  return(unlist(strsplit(paste(reduced.chars, collapse=''), '!')))
}


create.redundancy.vs.entropy.plot <- function(data) {
  ##
  #
  ##
  overall.redundancy <- sum(data$Correct_guesses)
  overall.entropy <- sum(data$Incorrect_guesses)
  total <- overall.redundancy + overall.entropy
  proportions <- c(overall.redundancy / total, overall.entropy / total)

  df <- data.frame(label=c('redundancy', 'entropy'), proportion=proportions)
  plot <- ggplot(df, aes(x=1, y=proportion, fill=factor(label)))
  plot <- plot + geom_bar(stat='identity', color='black') +  coord_polar(theta='y', start=0)
  plot <- plot + scale_fill_manual(values=c("red", "blue"))
  plot <- plot + theme(axis.ticks=element_blank(),  # the axis ticks
                       axis.title=element_blank(),  # the axis labels
                       axis.text.y=element_blank(), # the 0.75, 1.00, 1.25 labels
                       axis.text.x=element_blank())
  return(plot)
}


create.letter.guess.by.phrase.position.plot <- function(data) {
  ##
  #
  ##
  correct <- rep(0, 40)  # FIXME: use the real number, don't just guess 50
  phrase.results <- lapply(lapply(lapply(data$Reduced_text, FUN=strsplit, '*'), unlist), rev)
  for (result in phrase.results) {
    for (i in 1:length(result)) {
      letter <- result[i]
      if (letter == '-') {
        correct[i] = correct[i] + 1
      }
    }
  }

  df <- data.frame(reverse.index=0:39, correct)
  plot <- ggplot(df, aes(x=reverse.index, y=correct))
  plot <- plot + geom_bar(stat='identity')

  return(plot)
}


create.letter.guess.by.word.position.plot <-function(data) {
  correct <- rep(0, 9)  # FIXME: use the real number, don't just guess 25
  phrases <- data$Text
  reduced <- data$Reduced
  for (i in 1:length(phrases)) {
    reduced.words <- get.reduced.words(phrases[i], reduced[i])
    for (word in reduced.words) {
      word <- rev(unlist(strsplit(word, '*')))
      for (j in 1:length(word)) {
        if (word[j] == '-') {
          correct[j] = correct[j] + 1
        }
      }
    }
  }

  df <- data.frame(reverse.index=0:8, correct)
  plot <- ggplot(df, aes(x=reverse.index, y=correct))
  plot <- plot + geom_bar(stat='identity')
  plot <- plot + coord_cartesian(ylim = c(0, 25))

  return(plot)
}


examine.letter.guess.overall <- function(data) {
  ##
  #
  ##

  plot <- create.redundancy.vs.entropy.plot(data)
  plot <- plot + ggtitle('Overall Redundancy vs. Entropy')
  ggsave('overall-redundancy-entropy.png', plot=plot)

  plot <- create.letter.guess.by.phrase.position.plot(data)
  plot <- plot + ggtitle('Overall Correct Guesses by Phrase Position')
  ggsave('overall-correct-by-phrase-position.png', plot=plot)

  plot <- create.letter.guess.by.word.position.plot(data)
  plot <- plot + ggtitle('Overall Correct Guesses by Word Position')
  ggsave('overall-correct-by-word-position.png', plot=plot)

  return(0)
}


examine.letter.guess.by.participant <- function(data) {
  ##
  #
  ##

  participants <- as.list(levels(factor(data$Participant)))
  participant.frames <- list()
  for (participant in participants) {
    n <- subset(data, Participant == participant,
                select=c(Correct_guesses, Incorrect_guesses, Text, Reduced_text))
    participant.frames[[length(participant.frames) + 1]] <- n
  }

  # redundancy vs. entropy
  plots <- list()
  for (i in 1:length(participants)) {
    name <- participants[[i]]
    frame <- participant.frames[[i]]
    plot <- create.redundancy.vs.entropy.plot(frame)
    plots[[length(plots) + 1]] <- plot + ggtitle(name)
  }
  plot <- multiplot(plotlist=plots, cols=2)
  ggsave('participant-redundancy-entropy.png')

  # by phrase
  plots <- list()
  for (i in 1:length(participants)) {
    name <- participants[[i]]
    frame <- participant.frames[[i]]
    plot <- create.letter.guess.by.phrase.position.plot(frame)
    plots[[length(plots) + 1]] <- plot + ggtitle(name)
  }
  plot <- multiplot(plotlist=plots, cols=2)
  ggsave('participant-correct-by-phrase-position.png')

  # by word
  plots <- list()
  for (i in 1:length(participants)) {
    name <- participants[[i]]
    frame <- participant.frames[[i]]
    plot <- create.letter.guess.by.word.position.plot(frame)
    plots[[length(plots) + 1]] <- plot + ggtitle(name)
  }
  plot <- multiplot(plotlist=plots, cols=2)
  ggsave('participant-correct-by-word-position.png')

  return(0)
}


###
# 2-6
###

create.reaction.scatter.plot <- function(data, title='') {
  plot <- ggplot(data, aes(x=trial, y=reaction_time, color=Experiment, shape=Participant))
  plot <- plot + geom_point(size=3.5) + geom_point(color='black', size=1)
  plot <- plot + ggtitle(title)
  plot <- plot + coord_cartesian(ylim = c(150, 900))

  return(plot)
}


examine.reaction.data.overall <- function(data) {
  ##
  #
  ##


  plot <- create.reaction.scatter.plot(data, title='Overall Reaction Times (ms)')

  ggsave(plot, filename='overall-reaction.png')

  return(0)
}


examine.reaction.data.by.participant <- function(data) {
  ##
  #
  ##

  # data <- melted.reaction.data

  participants <- levels(factor(data$Participant))

  plots <- list()
  for (participant in participants) {
    sub.data <- subset(data, Participant == participant)
    plots[[length(plots) + 1]] <- create.reaction.scatter.plot(sub.data, title=participant)
  }

  multiplot(plotlist=plots, cols=2)

  return(0)
}


create.match.timing.plot <- function(data) {
  ##
  #
  ##
  plot <- ggplot(data=melted.matching.data, aes(x=Trial, y=reaction_time,
                                                color=Experiment, shape=Participant))
  plot <- plot + geom_point(size=2.0)
  plot <- plot + ggtitle('Matching Experiment Timings (ms)')
  return(plot)
}

create.errors.plot <- function(data) {
  ##
  #
  ##

  df <- aggregate(data$Errors, by=list(Participant=data$Participant), FUN=sum)

  plot <- ggplot(data=df, aes(x=Participant, y=x))
  plot <- plot + geom_bar(stat='identity')
  plot <- plot + scale_y_continuous(name='Number of Errors', limits=c(0,5))
  plot <- plot + ggtitle('Total Errors by Participant')

  return(plot)
}



main <- function() {
  # letter guess
  letter.guess.data <- read.csv(file=LETTER.GUESS.FILE, header=TRUE, stringsAsFactors=FALSE)
  examine.letter.guess.overall(letter.guess.data)
  examine.letter.guess.by.participant(letter.guess.data)


  # reaction
  full.reaction.data <- read.csv(file=REACTION.FILE, header=TRUE, stringsAsFactors=FALSE)
  extraneous.columns <- c('Block', 'Mode', 'mean', 'min', 'max', 'sd')
  reaction.data <- full.reaction.data[, !(names(full.reaction.data) %in% extraneous.columns)]

  melted.reaction.data <- melt(reaction.data, id=c('Experiment', 'Participant'),
                               variable.name='trial', value.name='reaction_time')

  examine.reaction.data.overall(melted.reaction.data)
  examine.reaction.data.by.participant(melted.reaction.data)


  # matching
  matching.data <- read.csv(file=MATCHING.FILE, header=TRUE, stringsAsFactors=FALSE)
  melted.matching.data <- melt(matching.data, id=c('Participant', 'Experiment', 'Errors'),
                               variable.name='Trial', value.name='reaction_time')

  plot <- create.match.timing.plot(melted.matching.data)
  ggsave('matching-timings.png', plot=plot)

  plot <- create.errors.plot(matching.data)
  ggsave('matching-errors.png', plot=plot)
}

main()

# print(getSrcDirectory(main))
