#!/usr/bin/env RScript

library(caret)
library(azuremlsdk)

args <- commandArgs(trailingOnly = T)

if (length(args) == 0) {
  print("Local environment: reading local file")
  all_data <- read.csv("./data/part-00000")
  save_path <- "./data/model"
} else {
  all_data <- read.csv(paste0(args[1], "/", "part-00000"))
  save_path <- args[2]
}


summary(all_data)

in_train <- createDataPartition(y = all_data$variety, p = .8, list = FALSE)
train_data <- all_data[in_train, ]
test_data <- all_data[-in_train, ]

# Run algorithms using 10-fold cross validation
control <- trainControl(method = "cv", number = 10)
metric <- "Accuracy"

set.seed(7)
model <- train(variety ~ .,
               data = train_data,
               method = "lda",
               metric = metric,
               trControl = control)
predictions <- predict(model, test_data)

conf_matrix <- confusionMatrix(predictions, as.factor(test_data$variety), mode="everything")
log_metric_to_run(metric, conf_matrix$overall["Accuracy"])

log_table_to_run("class_proportions",
                 as.list(prop.table(table(all_data$variety))))

ifelse(dir.exists(save_path), "dir exists", "creating dir for model")
dir.create(save_path, showWarnings = F, recursive = T)
saveRDS(model, file=paste0(save_path,"/model.rds"))

print("saved model!")
