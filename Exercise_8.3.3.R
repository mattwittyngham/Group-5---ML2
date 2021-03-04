# need some of the 8.32 code in order to have proper variables for 8.33 example
rm(list=ls())
library(MASS)
library(randomForest)
library(tree)
set.seed(1)
train = sample(1:nrow(Boston), nrow(Boston)/2)
tree.boston=tree(medv~.,Boston,subset = train)
summary(tree.boston)
# Regression Tree
tree(formula = medv~., data = Boston, subset = train)
plot(tree.boston)
text(tree.boston, pretty = 0)
cv.boston=cv.tree(tree.boston)
plot(cv.boston$size,cv.boston$dev,type = 'b')
# prune tree
prune.boston=prune.tree(tree.boston, best = 5)
plot(prune.boston)
text(prune.boston, pretty = 0)
# use unpruned tree to make predictions on the test set
yhat=predict(tree.boston, newdata = Boston[-train,])
boston.test=Boston[-train,"medv"]
plot(yhat,boston.test)
abline(0,1)
mean((yhat-boston.test)^2)



# 8.33 Bagging and Random Forests 
# in this example, we are applying the bagging and random forests methods to the Boston dataset
# to do this, we utilize the randomForest package in R

# bagging is just a special type of random forest with m=p, which means that 
# the randomForest() function can work for both

library(randomForest)
set.seed(1)
bag.boston=randomForest(medv~.,data = Boston, subset = train,
                        mtry=13, importance=TRUE)

# the argument mtry=13 lets us know that all 13 predictors should be considered
# for each split within the tree

bag.boston
## percentage of variance explained is 85.17%
## number of trees: 500
## number of variables tried at each split: 13

randomForest(formula = medv~., data = Boston, mtry = 13, 
             importance = TRUE, subset = train)
yhat.bag=predict(bag.boston, newdata = Boston[-train,])
plot(yhat.bag, boston.test)
abline(0,1)
mean((yhat.bag-boston.test)^2)

# the test set MSE associated with the bagged regression tree is 23.59273
# that is about 30% of that obtained using an optimally pruned single tree

# can change the number of trees grown by randromForest() by using the ntree argument

bag.boston=randomForest(medv~., data = Boston, subset = train,
                        mtry=13, ntree=25)

# this process is the same as above, except we are using a smaller value for the mtry argument

yhat.bag=predict(bag.boston,newdata = Boston[-train,])
mean((yhat.bag-boston.test)^2)

set.seed(1)
rf.boston=randomForest(medv~., data = Boston, subset = train,
                       mtry=6, importance=TRUE)
yhat.rf=predict(rf.boston, newdata = Boston[-train,])
mean((yhat.rf-boston.test)^2)

# the test set MSE is 19.62021; this tells us that random forests yielded an improved outcome
# over bagging in this case

# can use the importance() function to see the importance of each variable 
# this provides two measures of variable importance:
# the first is based on the mean decrease of accuracy in predictions on the out of bag samples
# when a given variable is excluded from the model
# The second is a measure of the total decrease in node impurity that comes from splits
# over that variable, averaged over all trees

# when dealing with regression trees, the node impurity is measured by the training RSS
# to plot these importance findings, you can use the varImpPlot() function

importance(rf.boston)
varImpPlot(rf.boston)

# the results show that across all of the tress considered in the random forest, the wealth 
# level of the community(lstat) amd the house size(rm) are far and away the two most important variables

## bagging typically gives us improved accuracy over prediction using just a single tree
# however, this accuracy can be achieved at the cost of ease of interpretation in some instances
# this means that often it is difficult to identify which variables are most important to the procedure
# the importance()function can be used as a tool to identify which variables are most important 
