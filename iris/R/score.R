#!/usr/bin/env RScript

library(azuremlsdk)

args <- commandArgs(trailingOnly = T)

if (length(args) == 0) {
  print("Local environment: reading local file")
  new.data <- read.csv("./data/iris-score.csv")
  model <- readRDS("./data/model.rds")
} else {
  model <- readRDS(paste0(args[1], "/model.rds"))
  new.data <- read.csv(args[2])
  print(model)
  print(head(new.data))
}

predictions <- predict(model, newdata = new.data)

predicted.df <- cbind(new.data, predictions)
colnames(predicted.df) <- c(colnames(new.data), "variety")

write.csv(predicted.df, "/tmp/scored.csv", row.names = F)



