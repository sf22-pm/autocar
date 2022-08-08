suppressPackageStartupMessages(library(arulesCBA))

cpar = function(train, test) {
  rdf_train <- train
  rdf_train$class <- as.factor(rdf_train$class)

  rdf_test <- test
  rdf_test$class <- as.factor(rdf_test$class)

  #capture.output(install_LUCS_KDD_CPAR(), file='NUL')

  clf <- CPAR(class ~ ., rdf_train, best_k = 1000)
  pred <- predict(clf, rdf_test)
  x <- as.integer(levels(pred))[pred]
  return(x)
}
