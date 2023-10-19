## ----setup, include=FALSE------------------------------------------------------------------------
knitr::opts_chunk$set(message = FALSE,
                      fig.show='hold',
                      fig.pos = "H", out.extra = "", echo=F, output=F)

options(digits = 3)

# Load packages if needed.
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-project.org")
if(!require(pROC)) install.packages("pROC", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(matrixStats) # colSds() function
library(pROC)
library(knitr)


## ------------------------------------------------------------------------------------------------
# Download data.
dl <- "ionosphere.zip"
if(!file.exists(dl))
  download.file("https://archive.ics.uci.edu/static/public/52/ionosphere.zip", dl)

data_file <- "ionosphere.data"
if(!file.exists(data_file))
  unzip(dl, data_file, exdir=".")


## ------------------------------------------------------------------------------------------------
# Load data.
d <- read.csv('ionosphere.data', header=F)
y <- d[length(d)]
colnames(y) <- c("y")
d <- d[1:34]


## ---- fig.cap="Diagram of the ionosphere (Guest 2003).", fig.align="center", out.width="55%"-----
knitr::include_graphics("figures/ionosphere-diagram.png")


## ------------------------------------------------------------------------------------------------
# Convert outcomes into 1 and 0.
y <- ifelse(y == "g", 1, 0)


## ------------------------------------------------------------------------------------------------
# Remove any columns having a standard deviation of zero.
zero_cols <- which(colSds(as.matrix(d)) == 0)
d <- d[-zero_cols]
rm(zero_cols)


## ------------------------------------------------------------------------------------------------
# Calculate percentages of outcomes.
perc_good <- mean(y == 'g')
perc_bad <- mean(y == 'b')


## ---- fig.cap="Heatmap of ACF data; green indicates good; pink indicates bad.", fig.align='center'----
# Generate heatmap of data.
heatmap(as.matrix(d),
        RowSideColors = ifelse(y == 1, "green", "pink"))


## ------------------------------------------------------------------------------------------------
# Scale and perform PCA.
d_scaled <- scale(d, center=T, scale=T)
pca <- prcomp(d_scaled, scale.=F)


## ---- fig.cap="Plot of principal components 1 and 2.", out.width="70%", fig.align='center'-------
# Plot PC1 vs PC2.
data.frame(pca$x[,1:2], rslt=y) |>
   ggplot(aes(PC1, PC2, fill=y)) +
   geom_point(cex=3, pch=21) +
   coord_fixed(ratio=1)


## ---- fig.cap="Plot of principal components 2 and 3.", out.width="70%", fig.align='center'-------
# Plot PC2 vs PC3.
data.frame(pca$x[,2:3], rslt=y) |>
   ggplot(aes(PC2, PC3, fill=y)) +
   geom_point(cex=3, pch=21) +
   coord_fixed(ratio=1)


## ---- fig.cap="Plot of principal components 1 and 3.", out.width="70%", fig.align='center', fig.show='hold'----
# Plot PC1 vs PC3.
data.frame(pca$x[,1:3], rslt=y) |>
  ggplot(aes(PC1, PC3, fill=y)) +
    geom_point(cex=3, pch=21) +
    coord_fixed(ratio=1)


## ---- warning=FALSE------------------------------------------------------------------------------
# Display table with data on first three PCs.
first_3_pcs <- as_tibble(
  cbind(c('Standard deviation', 'Proportion of Variance', 'Cumulative Proportion'),
  summary(pca)$importance))[1:4]
colnames(first_3_pcs) <- c("Measure", "PC1", "PC2", "PC3")
knitr::kable(first_3_pcs)


## ---- fig.cap="Proportion of variance by PC.", fig.align='center', out.width="70%"---------------
# Proportion of variance plot.
plot(pca$sdev^2/sum(pca$sdev^2), xlab="PC", ylab="Variance proportion", type="b", col="orange")


## ------------------------------------------------------------------------------------------------
# Split the data into working and holdout sets.
set.seed(17)
holdout_idx <- createDataPartition(y, times = 1, p = 0.25, list=F)
holdout_set <- d[holdout_idx,] 
y_holdout <- y[holdout_idx]
working_set <- d[-holdout_idx,]
y_working <- y[-holdout_idx]
# Further split working into training and validation sets.
validate_idx <- createDataPartition(y_working, times=1, p=0.33, list=F)
validate_set <- working_set[validate_idx,]
y_validate <- y_working[validate_idx]
train_set <- working_set[-validate_idx,]
y_train <- y_working[-validate_idx]
rm(holdout_idx)
rm(validate_idx)


## ---- fig.cap="Distribution of the response variable in the training set.", out.width="50%", fig.align='center'----
# Plot proportions of outcomes in the train set.
y_train |> as_tibble() |>
  ggplot(aes(y_train)) +
  geom_bar() +
  scale_x_continuous(breaks=0:1) +
  xlab("ACF Quality (0=Clutter, 1=Good)") +
  ylab("Tally")


## ------------------------------------------------------------------------------------------------
# Set up CV config for reuse.
tr_control <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10)


## ---- warning=FALSE------------------------------------------------------------------------------
# Fit linear regression model and generate predictions.
fit_lr <- train(train_set, y_train, method="lm", trControl = tr_control)
y_hat_lr_cont <- predict(fit_lr, validate_set)
y_hat_lr <- ifelse(y_hat_lr_cont > 0.5, 1, 0)


## ------------------------------------------------------------------------------------------------
# Determine statistics for LR.
cm_lr <- confusionMatrix(as.factor(y_hat_lr), as.factor(y_validate), positive="1")
outcome_lr <- c(cm_lr$overall['Accuracy'], 
                cm_lr$byClass[c('Sensitivity', 'Specificity', "F1")])


## ------------------------------------------------------------------------------------------------
# Display results for LR.
kable(data.frame(outcome_lr), col.names = c("Results for linear regression (validation set)."))


## ---- fig.cap="ROC curve for LR.", fig.align='center', out.width="60%"---------------------------
# ROC curve for LR.
roc_lr <- roc(y_validate, y_hat_lr)
auc_lr <- round(auc(y_validate, y_hat_lr),4)
ggroc(roc_lr,  colour = 'orange', size = 2) +
  ggtitle(paste0('(AUC = ', auc_lr, ')'))


## ---- warning=FALSE------------------------------------------------------------------------------
# Fit KNN and generate predictions.
set.seed(17)
fit_knn <- train(train_set, y_train, method="knn", 
                 trControl = tr_control,
                 tuneGrid=data.frame(k=seq(3,13)))
best_k <- fit_knn$bestTune |> pull(k)
y_hat_knn_cont <- predict(fit_knn, validate_set)
y_hat_knn <- ifelse(y_hat_knn_cont > 0.5, 1, 0)
# Determine statistics for KNN.
cm_knn <- confusionMatrix(as.factor(y_hat_knn), as.factor(y_validate), positive="1")
outcome_knn <- c(cm_knn$overall['Accuracy'], 
                cm_knn$byClass[c('Sensitivity', 'Specificity', "F1")])


## ------------------------------------------------------------------------------------------------
# Display table of results for KNN.
kable(data.frame(outcome_knn), col.names = c("Results for KNN (validation set)."))


## ---- fig.cap="ROC curve for KNN.", fig.align='center', out.width="60%"--------------------------
# ROC curve for KNN.
roc_knn <- roc(y_validate, y_hat_knn)
auc_knn <- round(auc(y_validate, y_hat_knn),4)
ggroc(roc_knn,  colour = 'orange', linetype=1, size = 2) +
  ggtitle(paste0('(AUC = ', auc_knn, ')'))


## ---- warning=FALSE------------------------------------------------------------------------------
# Fit RF and generate predictions.
set.seed(17)
fit_rf <- train(train_set, y_train, method="rf",
                trControl = tr_control,
                tuneGrid=data.frame(mtry=5:11))
best_mtry <- fit_rf$bestTune |> pull(mtry)
y_hat_rf_cont <- predict(fit_rf, validate_set)
y_hat_rf <- ifelse(y_hat_rf_cont > 0.5, 1, 0)


## ------------------------------------------------------------------------------------------------
# Determine statistics for random forest.
cm_rf <- confusionMatrix(as.factor(y_hat_rf), as.factor(y_validate), positive="1")
outcome_rf <- c(cm_rf$overall['Accuracy'], 
                cm_rf$byClass[c('Sensitivity', 'Specificity', "F1")])


## ------------------------------------------------------------------------------------------------
# Display results for random forest.
kable(data.frame(outcome_rf), col.names = c("Results for random forest (validation set)."))


## ---- fig.cap="ROC curve for random forest.", fig.align='center', out.width="60%"----------------
# ROC curve for RF.
roc_rf <- roc(y_validate, y_hat_rf)
auc_rf <- round(auc(y_validate, y_hat_rf),4)
ggroc(roc_rf,  colour = 'orange', linetype=1, size = 2) +
  ggtitle(paste0('(AUC = ', auc_rf, ')'))


## ------------------------------------------------------------------------------------------------
# Generate ensemble predictions.
y_hat_ensemble_cont <- (y_hat_lr_cont + y_hat_knn_cont + y_hat_rf_cont) / 3
y_hat_ensemble <- ifelse(y_hat_ensemble_cont > 0.5, 1, 0)


## ------------------------------------------------------------------------------------------------
# Determine statistics for ensemble.
cm_ens <- confusionMatrix(as.factor(y_hat_ensemble), as.factor(y_validate), positive="1")
outcome_ens <- c(cm_ens$overall['Accuracy'], 
                cm_ens$byClass[c('Sensitivity', 'Specificity', "F1")])


## ------------------------------------------------------------------------------------------------
# Display results for ensemble.
kable(data.frame(outcome_ens), col.names = c("Results for ensemble model (validation set)."))


## ---- fig.cap="ROC curve for ensemble.", fig.align='center', out.width="60%"---------------------
# ROC curve for ensemble model.
roc_ens <- roc(y_validate, y_hat_ensemble)
auc_ens <- round(auc(y_validate, y_hat_ensemble),4)
ggroc(roc_ens,  colour = 'orange', linetype=1, size = 2) +
  ggtitle(paste0('(AUC = ', auc_ens, ')'))


## ------------------------------------------------------------------------------------------------
# Ensemble with holdout; need to build out step by step.

y_hat_lr_cont_hout <- predict(fit_lr, holdout_set)
y_hat_lr_hout <- ifelse(y_hat_lr_cont_hout > 0.5, 1, 0)

y_hat_knn_cont_hout <- predict(fit_knn, holdout_set)
y_hat_knn_hout <- ifelse(y_hat_knn_cont_hout > 0.5, 1, 0)

y_hat_rf_cont_hout <- predict(fit_rf, holdout_set)
y_hat_rf_hout <- ifelse(y_hat_rf_cont_hout > 0.5, 1, 0)

cm_rf_final <- confusionMatrix(as.factor(y_hat_rf_hout),
                            as.factor(y_holdout),
                            positive="1")
outcome_rf_final <- c(cm_rf_final$overall['Accuracy'], 
                   cm_rf_final$byClass[c('Sensitivity', 'Specificity', "F1")])

y_hat_ensemble_cont_hout <- (y_hat_lr_cont_hout + y_hat_knn_cont_hout + y_hat_rf_cont_hout) / 3
y_hat_ensemble_hout <- ifelse(y_hat_ensemble_cont_hout > 0.5, 1, 0)

cm_final <- confusionMatrix(as.factor(y_hat_ensemble_hout),
                            as.factor(y_holdout),
                            positive="1")
outcome_final <- c(cm_final$overall['Accuracy'], 
                   cm_final$byClass[c('Sensitivity', 'Specificity', "F1")])


## ---- fig.show='hold', fig.cap="ROC curve for ensemble and RF (using holdout set).", fig.align='center', out.width="49%"----
# ROC curve for ensemble model w/ holdout.
roc_ens_h <- roc(y_holdout, y_hat_ensemble_hout)
auc_ens_h <- round(auc(y_holdout, y_hat_ensemble_hout),4)
ggroc(roc_ens_h,  colour = 'orange', linetype=1, size = 2) +
  ggtitle(paste0('Ensemble with holdout data set (AUC = ', auc_ens_h, ')'))

# ROC curve for RF w/ holdout.
roc_rf_h <- roc(y_holdout, y_hat_rf_hout)
auc_rf_h <- round(auc(y_holdout, y_hat_rf_hout),4)
ggroc(roc_rf_h,  colour = 'orange', linetype=1, size = 2) +
  ggtitle(paste0('RF with holdout data set (AUC = ', auc_rf_h, ')'))


## ------------------------------------------------------------------------------------------------
# Display table of all outcomes.
all_outcomes <- cbind(outcome_lr, outcome_knn, outcome_rf, outcome_ens, 
                      outcome_rf_final, outcome_final)
colnames(all_outcomes) <- c("LR", "KNN", "RF", "Ensemble", "RF_holdout", "Ensemble_holdout")
kable(data.frame(all_outcomes))

