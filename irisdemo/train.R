# This script loads a dataset of which the last column is supposed to be the
# class and logs the accuracy

library(azuremlsdk)
library(caret)

all_data <- read.csv("iris.csv")
summary(all_data)

in_train <- createDataPartition(y = all_data$Species, p = .8, list = FALSE)
train_data <- all_data[in_train, ]
test_data <- all_data[-in_train, ]

# Run algorithms using 10-fold cross validation
control <- trainControl(method = "cv", number = 10)
metric <- "Accuracy"

set.seed(7)
model <- train(Species ~ .,
               data = train_data,
               method = "lda",
               metric = metric,
               trControl = control)
predictions <- predict(model, test_data)
conf_matrix <- confusionMatrix(predictions, test_data$Species, mode="everything")
message(conf_matrix)

log_metric_to_run(metric, conf_matrix$overall["Accuracy"])
log_metric_to_run("F1-Score", conf_matrix$overall["F1"])
log_metric_to_run("Precision", conf_matrix$overall["Precision"])
log_metric_to_run("Recall", conf_matrix$overall["Recall"])

saveRDS(model, file="./outputs/model_trained.rds")

ws <- load_workspace_from_config()

#Register Model to the
model <- register_model(ws, 
                        model_path = "outputs/model_trained.rds", 
                        model_name = "iris_model_trained",
                        description = "Predict class of the Iris flower")

