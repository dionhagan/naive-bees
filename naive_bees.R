# SCRIPT TO LOAD TRAIN/TEST DATA INTO DATA FRAME
library(jpeg)
library(plyr)

#remove image data if exists
if (exists("traindf")) {
  rm(traindf)
}
if (exists("testdf")) {
  rm(testdf)
}

train.files = list.files(path="./images/train/")
test.files = list.files(path="./images/test/")
#temp = list.files(path="./images/temp")

#import training data labels
labels = read.csv("./train_labels.csv")
labels = arrange(labels, labels$id)
labels$genus = as.factor(labels$genus)

#length of each list
train_cnt <- length(train.files)
test_cnt <- length(test.files)

#flatten to wide data: (m, n) -> (1, m*n)
flatten_image = function(img) {
  img = c(img)
  img
}

#split training images into train & test sets
smpl = sample(train.files, 1500, replace=FALSE)
train = smpl[1:1200]
test = smpl[1201:1500]

#load sample training images (imperative)
for (file in train)
{
  dir = paste("./images/train/", file, sep="")
  img = readJPEG(dir, native=TRUE)
  img = flatten_image(img)
  tempdf = data.frame(img)
  #transpose column to row:  (1, m) -> m
  newrow = t(tempdf)
  #initialize image dataframe
  if (!exists("traindf")) {
    traindf = newrow
  }
  else {
    #append new row to df
    traindf = rbind(traindf, newrow)
  }
}
"Train Images imported"

#load sample testing images (imperative)
for (file in test)
{
  dir = paste("./images/train/", file, sep="")
  img = readJPEG(dir, native=TRUE)
  img = flatten_image(img)
  tempdf = data.frame(img)
  #transpose column to row:  (1, m) -> m
  newrow = t(tempdf)
  #initialize image dataframe
  if (!exists("testdf")) {
    testdf = newrow
  }
  else {
    #append new row to df
    testdf = rbind(testdf, newrow)
  }
}
"Test Images imported"

#dimensions of train/test data
#partially serves to mark end of loop
cat(sprintf("Training set has %d rows and %d columns\n", nrow(traindf), ncol(traindf)))
cat(sprintf("Test set has %d rows and %d columns\n", nrow(testdf), ncol(testdf)))

#extract number from file name
train.num = sapply(train, 
                   FUN = function(x) as.numeric(gsub(".jpg", "", x)))
test.num = sapply(test, 
                  FUN = function(x) as.numeric(gsub(".jpg", "", x)))

#set row names to image id
dimnames(traindf)[[1]] = unname(train.num)
dimnames(testdf)[[1]] = unname(test.num)

#attach labels to both df's
trainrows = dimnames(traindf)[1]
trainrows = sapply(trainrows, FUN = function(x) as.numeric(x))
traindf = traindf[order(trainrows),]

testrows = dimnames(testdf)[1]
testrows = sapply(testrows, FUN = function(x) as.numeric(x))
testdf = testdf[order(testrows),]

train.labels = labels[labels$id %in% trainrows,]
test.labels = labels[labels$id %in% testrows,]

labeldf.trn = cbind(train.labels$id, train.labels$genus)
labeldf.tst = cbind(test.labels$id, test.labels$genus)

# Random PCA to reduce dimensionality
pca = prcomp(traindf[,1:40000],
             center = TRUE,
             scale. = TRUE)

#screeplot to check number of components to extract
screeplot(pca, type="lines")

#percent variance explained by components
pve = pca$sdev^2/sum(pca$sdev^2)
plot(pve, pch = 19, type = "l")

#How many PCs contains a cumulative pve > 0.9
min(which(cumsum(pve) > 0.9))
cumsum(pve)[10]
cumsum(pve)[100]
cumsum(pve)[250]

#apply pca to df's
train.pca <- traindf[,1:40000] %*% pca$rotation
test.pca <- testdf %*% pca$rotation

#get data ready for classifier
Xtrain <- train.pca[,1:10]
Xtest <- test.pca[,1:10]
ytrain = train.labels$genus
ytest = test.labels$genus

library(caret)
library(klaR)
library(e1071)

#get accuracy function
accuracy_score <- function(pred, y)
{
  correct = 0
  for(i in 1:length(pred))
  {
    if(pred[i] == y[i])
    {
      correct = correct + 1 
    }
  }
  acc = correct/length(pred)
}

#configure CV
fitControl = trainControl(#10-fold Cross-Validation
  method='repeatedCV', 
  number=10, 
  #repeat CV 10x
  repeats=10)



# NAIVE BAYES
t0 <- Sys.time()
nb = train(Xtrain, ytrain,'nb',
                 trControl=fitControl,
                 verbose=FALSE)
print(Sys.time() - t0)
#predict test data
predictions.test = predict(nb, Xtest)
#calculate accuracy scores
score.test = accuracy_score(predictions.test, ytest)
cat(sprintf("Training Accuracy: %0.02f", nb$results$Accuracy[2]))
cat(sprintf("Testing Accuracy: %0.02f", score.test))
confusionMatrix(predictions.test, ytest)
# Train: 0.78
# Test:  0.76

# RANDOM FOREST
rf = train(Xtrain, ytrain,'rf',
                 trControl=fitControl,
                 verbose=FALSE)
#predict test data
predictions.test = predict(rf, Xtest)
#calculate accuracy scores
score.test = accuracy_score(predictions.test, ytest)
cat(sprintf("Training Accuracy: %0.02f", rf$results$Accuracy[2]))
cat(sprintf("Testing Accuracy: %0.02f", score.test))
confusionMatrix(predictions.test, ytest)
# Train: 0.79
# Test:  0.75

#ADABOOST
ada = train(Xtrain, ytrain,'ada',
                 trControl=fitControl,
                 verbose=FALSE)
#predict test data
predictions.test = predict(ada, Xtest)
#calculate accuracy scores
score.test = accuracy_score(predictions.test, ytest)
cat(sprintf("Training Accuracy: %0.02f", ada$results$Accuracy[2]))
cat(sprintf("Testing Accuracy: %0.02f", score.test))
confusionMatrix(predictions.test, ytest)
# Train: 0.79
# Test:  0.76
