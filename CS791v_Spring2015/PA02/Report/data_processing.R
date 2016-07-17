setwd("~/Desktop/CS791v_Spring2014/PA02/Report")

results.table <- read.csv("results.csv")
# results.table <- subset(results.table, subset = results.table$Vector.Size %% 1000000 == 0)
results.table <- data.frame(
  results.table,
  Compute.Speedup = rep(0, nrow(results.table)),
  Total.Speedup = rep(0, nrow(results.table))
)

devices <- levels(results.table$Device.Type)
vector.sizes <- lapply(levels(as.factor(results.table$Vector.Size)), as.numeric)

for (i in 1:nrow(results.table)) {
  corresponding.sequential.trial <- subset(
    results.table,
    results.table$Vector.Size == results.table$Vector.Size[[i]] &
      results.table$Device.Type == devices[[2]]
  )
  if (nrow(corresponding.sequential.trial) > 1) {
    corresponding.sequential.trial <- corresponding.sequential.trial[1, ]
  }
  
  results.table$Compute.Speedup[[i]] <-
    corresponding.sequential.trial$Compute.Time..s. /
      results.table$Compute.Time..s.[[i]]

  results.table$Total.Speedup[[i]] <-
    corresponding.sequential.trial$Compute.Time..s. /
    results.table$Total.Time..s.[[i]]
}

#(results.table[order(-results.table$Compute.Speedup), ])[1:10, ]
#(results.table[order(-results.table$Compute.Speedup, -results.table$Vector.Size), ])[1:20, c(1,2,8,9,11)]
# use 65535 blocks and 1024 threads, it has a lot of the good speedups, for one vector use 4500000
ideal.threads.blocks <- subset(results.table, results.table$Number.of.GPU.Blocks == 63 & results.table$Number.of.Threads.per.Block == 256)
ideal.threads.blocks.plottable <- data.frame(
  "Device" = ideal.threads.blocks$Device.Type,
  "Vector Size" = ideal.threads.blocks$Vector.Size,
  "Blocks" = rep(63, nrow(ideal.threads.blocks)),
  "Threads" = rep(256, nrow(ideal.threads.blocks)),
  "Compute Time (s)" = ideal.threads.blocks$Compute.Time..s.,
  "Total [Compute + Xfer] Time (s)" = ideal.threads.blocks$Total.Time..s.,
  "Compute Throughput (int/s)" = ideal.threads.blocks$Compute.Time..s.,
  "Total [Compute + Xfer] Throughput (int/s)" = ideal.threads.blocks$Total.Time..s.
)
write.table(ideal.threads.blocks.plottable, file = "block_thread.csv", sep = ',')

library(ggplot2)
library(reshape2)
melt(ideal.threads.blocks.plottable)
ggplot(
  data= ideal.threads.blocks.plottable,
  aes(
    x = "Vector Size",
    y = "Runtime (s)",
    group = "Type",
   )) + geom_line() + geom_point()


good.size <- subset(results.table, results.table$Vector.Size == 14000000)
good.size.plottable <- data.frame(
  "Device" = good.size$Device.Type,
  "Vector Size" = good.size$Vector.Size,
  "Blocks" = good.size$Number.of.GPU.Blocks,
  "Threads" = good.size$Number.of.Threads.per.Block,
  "Compute Time (s)" = good.size$Compute.Time..s.,
  "Total [Compute + Xfer] Time (s)" = good.size$Total.Time..s.,
  "Compute Throughput (int/s)" = good.size$Compute.Time..s.,
  "Total [Compute + Xfer] Throughput (int/s)" = good.size$Total.Time..s.,
  "Speedup" = good.size$Compute.Speedup
)
write.table(good.size.plottable, file = "good_size.csv", sep = ',')





