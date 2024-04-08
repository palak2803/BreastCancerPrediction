library(caret)
library(randomForest)
library(e1071) # For SVM
library(class)
library(kknn)

# Load the dataset
data_temp <- read.csv("wdbc.csv", stringsAsFactors = TRUE)

# Convert diagnosis to a binary factor
data_temp$diagnosis <- as.factor(ifelse(data_temp$diagnosis == "M", 1, 0))

data <- data_temp[, c("radius_mean", "texture_mean", "perimeter_mean", "area_mean", 
                  "smoothness_mean", "compactness_mean","concavity_mean", "concave_points_mean","symmetry_mean","fractal_dimension_mean","diagnosis")]
write.csv(data, "data_final.csv", row.names = FALSE)

# Extracting predictors and target
predictors <- data[, c("radius_mean", "texture_mean", "perimeter_mean", "area_mean", 
                       "smoothness_mean", "compactness_mean","concavity_mean", "concave_points_mean","symmetry_mean","fractal_dimension_mean")]
target <- data$diagnosis


# Split the dataset into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(target, p = .8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Prepare the training and test data for modeling
trainX <- trainData[, names(trainData) %in% names(predictors)]
trainY <- trainData$diagnosis
testX <- testData[, names(testData) %in% names(predictors)]
testY <- testData$diagnosis


# Train Logistic Regression Model
logisticModel <- glm(diagnosis ~ ., data = cbind(trainX, diagnosis = trainY), family = "binomial")

# Train SVM Model
svmModel <- svm(diagnosis ~ ., data = cbind(trainX, diagnosis = trainY), type = 'C-classification', kernel = 'linear')

# Train Random Forest Model
randomForestModel <- randomForest(diagnosis ~ ., data = cbind(trainX, diagnosis = trainY), importance=TRUE,ntree = 500)
print(randomForestModel$importance)


# Train KNN Model
knnModel <- knn3(diagnosis ~ ., data = cbind(trainX, diagnosis = trainY), k = 5)


# Calculate accuracies
logisticPreds <- ifelse(predict(logisticModel, newdata = testX, type = "response") > 0.5, 1, 0)
logisticAccuracy <- mean(logisticPreds == testY)

svmPreds <- predict(svmModel, newdata = testX, type = "response")
svmAccuracy <- mean(svmPreds == testY)

randomForestPreds <- predict(randomForestModel, newdata = testX)
randomForestAccuracy <- mean(randomForestPreds == testY)

knnPreds <- predict(knnModel, testX)
knnPreds <- ifelse(knnPreds[,2] > 0.5, 1, 0)
knnAccuracy <- mean(knnPreds == testY)

# Save accuracies
accuracies <- c(logistic = logisticAccuracy, svm = svmAccuracy, 
                randomForest = randomForestAccuracy, knn = knnAccuracy)


testData$diagnosis <- as.factor(testData$diagnosis)
logisticPreds <- as.factor(logisticPreds)
svmPreds <- as.factor(svmPreds)
randomForestPreds <- as.factor(randomForestPreds)
knnPreds <- as.factor(knnPreds)


# Calculate and print confusion matrices
logisticCM <- confusionMatrix(logisticPreds, testData$diagnosis)
svmCM <- confusionMatrix(svmPreds, testData$diagnosis)
randomForestCM <- confusionMatrix(randomForestPreds, testData$diagnosis)
knnCM <- confusionMatrix(knnPreds, testData$diagnosis)


# Save the models to disk
saveRDS(logisticModel, "logisticModel.rds")
saveRDS(svmModel, "svmModel.rds")
saveRDS(randomForestModel, "randomForestModel.rds")
saveRDS(knnModel, "knnModel.rds")
saveRDS(accuracies, "modelAccuracies.rds")
saveRDS(list(logistic=logisticCM, svm=svmCM, randomForest=randomForestCM, knn=knnCM), "modelMetrics.rds")

data_final <- data

df_structure <- data.frame(
  Feature_Name = c("ID","radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
                   "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean","diagnosis"),
  Description = c("Case Identifier","Average of distances from center to points on the perimeter",
                  "Standard deviation of gray-scale values",
                  "Mean size of the core tumor",
                  "Area of the tumor",
                  "Local variation in radius lengths",
                  "Severity of concave portions of the contour",
                  "Number of concave portions of the contour",
                  "Mean of the number of concave points",
                  "Symmetry of the tumor",
                  "Complexity of the tumor boundary","Malignant or Benign Tumor"),
  Datatype = c("Integer","numeric", "numeric", "numeric", "numeric", "numeric",
               "numeric", "numeric", "numeric", "numeric", "numeric","Categorical")
)

# Print the dataframe to view its structure

calculateSummaryStats <- function(data, selectedVars) {
  selectedData <- data[selectedVars]
  stats <- data.frame(
    Feature = names(selectedData),
    Mean = sapply(selectedData, mean, na.rm = TRUE),
    Median = sapply(selectedData, median, na.rm = TRUE),
    SD = sapply(selectedData, sd, na.rm = TRUE),
    Min = sapply(selectedData, min, na.rm = TRUE),
    Max = sapply(selectedData, max, na.rm = TRUE),
    stringsAsFactors = FALSE  # Ensure strings are not converted to factors
  )
  print('###################')
  return(stats)
}

