##install.packages("https://github.com/jeremiedb/mxnet_winbin/raw/master/mxnet.zip", repos = NULL)
library(caret)
library(mxnet)


#Importing the dataset
train <- data.matrix(read.csv("C://Users//Downloads//PROJECT CODINGS//mnist_train.csv", header=T))
dim(train[,-1])
test <- data.matrix(read.csv("C://Users//Downloads//PROJECT CODINGS//mnist_test.csv", header=T))
dim(test[,-1])
########################################################################


### Read the data

barplot(table(train[,1]),col = rainbow(10,0.5),main = "Digits in train")

plotTrain <- function(image){
  op <-par(no.readonly = TRUE)
  x<-ceiling(sqrt(length(image)))
  par(mfrow=c(x,x),mar=c(.1, .1, .1, .1))
  
  for(i in image){
    m<-matrix(train[i,-1], nrow = 28, byrow = TRUE)
    m <- apply(m, 2, rev)
    image(t(m),col = gray.colors(255),axes = FALSE)
    #text(0.05,0.2,col="white",cex = 1.2,train[i,1])
  }
  par(op)
  
}

plotTrain(1:100)

### Prepare data

train.x <- train[,-1]
train.y <-train[,1]
train.x <-t(train.x/255)
test.x <- t(test[,-1]/255)
test.y<-test[,1]
head(train.x)

### building of Simple neural Network

m1.data <- mx.symbol.Variable("data") # Notice how each layer is passed to the next 

m1.fc1 <- mx.symbol.FullyConnected(m1.data, name="fc1", num_hidden=128)
m1.act1 <- mx.symbol.Activation(m1.fc1, name="activation1", act_type="relu")

m1.fc2 <- mx.symbol.FullyConnected(m1.act1, name="fc2", num_hidden=64)
m1.act2 <- mx.symbol.Activation(m1.fc2, name="activation2", act_type="relu")

m1.fc3 <- mx.symbol.FullyConnected(m1.act2, name="fc3", num_hidden=10)
m1.softmax <- mx.symbol.SoftmaxOutput(m1.fc3, name="softMax")


log <- mx.metric.logger$new() #to keep track of the results each iterration
tick <- proc.time() #mark the start time
mx.set.seed(0)


m1 <- mx.model.FeedForward.create(m1.softmax,  #the network configuration made above
                                  X = train.x, #the predictors
                                  y = train.y, #the labels
                                  #ctx = mx.cpu(),
                                  num.round = 30, # The kernel can only handle 1 (I suggest ~50ish to start)
                                  array.batch.size = 100,
                                  array.layout="colmajor",
                                  learning.rate = 0.001,
                                  momentum = 0.95,
                                  eval.metric = mx.metric.accuracy,
                                  initializer = mx.init.uniform(0.07),
                                  epoch.end.callback = mx.callback.log.train.metric(1,log)
)

print(paste("Training took:", round((proc.time() - tick)[3],2),"seconds"))
plot(log$train, type="l", col="red", xlab="Iteration", ylab="Accuracy")



### Making Prediction

m1.preds <- predict(m1, test.x, array.layout = "colmajor")
t(round(m1.preds[,1:10], 2))

m1.preds.value <- max.col(t(m1.preds)) - 1
m1.preds.value[1:10]

### Checking Accuracy

Accuracy(m1.preds.value, test.y)

plotResults <- function(images, preds){
  op <- par(no.readonly=TRUE)
  x <- ceiling(sqrt(length(images)))
  par(mfrow=c(x,x), mar=c(.1, .1, .1, .1))
  
  for (i in images){
    m <- matrix(test[i,-1], nrow=28, byrow=TRUE)
    m <- apply(m, 2, rev)
    image(t(m), col=grey.colors(255), axes=FALSE)
    text(0.05,0.1,col="white", cex=1.2, preds[i])
  }
  par(op)
}

plotResults(1:64, m1.preds.value)
