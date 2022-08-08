suppressPackageStartupMessages(library(arulesCBA))

cba = function(train, test, s, c, l) {
  rdf_train <- train
  rdf_train$class <- as.factor(rdf_train$class)

  rdf_test <- test
  rdf_test$class <- as.factor(rdf_test$class)

  clf <- CBA(class ~ ., data = rdf_train , supp = s, conf = c, parameter = list(minlen = 2, maxlen = l), maxtime = 600)
  pred <- predict(clf, rdf_test)
  #print(pred)
  x <- as.integer(levels(pred))[pred]
  return(x)
}
