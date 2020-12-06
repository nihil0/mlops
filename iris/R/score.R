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

ifelse(dir.exists(save_path), "dir exists", "creating dir for model")
dir.create("./plots", showWarnings = F, recursive = T)
png("./plots/class_props.png")
plot(
  prop.table(table(predicted.df$variety)),
  xlab = "Variety",
  ylab = "Proportion"
)
dev.off()
log_image_to_run("ClassProportions", "./plots/class_props.png")

write.csv(predicted.df, "/tmp/scored.csv", row.names = F)



