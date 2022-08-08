suppressPackageStartupMessages(library(arulesCBA))

cmar = function(train, test, s, c) {
  rdf_train <- train
  rdf_train$class <- as.factor(rdf_train$class)

  rdf_test <- test
  rdf_test$class <- as.factor(rdf_test$class)

  #capture.output(install_LUCS_KDD_CMAR(), file='NUL')

  clf <- CMAR(class ~ ., data = rdf_train , support = s, confidence = c)
  pred <- predict(clf, rdf_test)
  x <- as.integer(levels(pred))[pred]
  return(x)
}
