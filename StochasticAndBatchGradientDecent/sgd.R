
##Problem 0. Data Preprocessing

#load train and test data
train <- read.csv('mnist_train.csv', header = FALSE)
test <- read.csv('mnist_test.csv', header = FALSE)

#partition to 0&1,3&5
train_0_1=train[,train[785,]==0|train[785,]==1]
test_0_1=test[,test[785,]==0|test[785,]==1]

train_3_5=train[,train[785,]==3|train[785,]==5]
test_3_5=test[,test[785,]==3|test[785,]==5]

#separate label row and store in 
true_label_train_0_1=as.numeric(train_0_1[785,])
true_label_test_0_1=as.numeric(test_0_1[785,])

true_label_train_3_5=as.numeric(train_3_5[785,])
true_label_test_3_5=as.numeric(test_3_5[785,])

#remove label row
train_0_1=train_0_1[-785,]
test_0_1=test_0_1[-785,]

train_3_5=train_3_5[-785,]
test_3_5=test_3_5[-785,]

#Visualize 1 image from each class
#plot them in a same picture
par(mfrow=c(2,2))

#sample one column for class 0 to plot in train_0_1

selectTr0=sample(which(true_label_train_0_1==0),1)
str0=matrix(train_0_1[,selectTr0],ncol = 28)
str0=t(str0[28:1,])
trueLabTr0=true_label_train_0_1[selectTr0]
image(str0,col=gray.colors(256), xlab = trueLabTr0,main="Class 0 from train_0_1")

#sample one column for class 1 to plot in test_0_1
selectTe1=sample(which(true_label_test_0_1==1),1)
ste1=matrix(test_0_1[,selectTe1],ncol = 28)
ste1=t(ste1[28:1,])
trueLabTe1=true_label_test_0_1[selectTe1]
image(ste1,col=gray.colors(256),xlab = trueLabTe1,main="Class 1 from test_0_1")

#sample one column for class 3 to plot in train_3_5
selectTr3=sample(which(true_label_train_3_5==3),1)
str3=matrix(train_3_5[,selectTr3],ncol = 28)
str3=t(str3[28:1,])
trueLabTr3=true_label_train_3_5[selectTr3]
image(str3,col=gray.colors(256),xlab = trueLabTr3,main="Class 3 from train_3_5")

#sample one column for class 5 to plot in test_3_5
selectTe5=sample(which(true_label_test_3_5==5),1)
ste5=matrix(test_3_5[,selectTe5],ncol = 28)
ste5=t(ste5[28:1,])
trueLabTe5=true_label_test_3_5[selectTe5]
image(ste5,col=gray.colors(256),xlab = trueLabTe5,main="Class 5 from test_3_5")

##Problem 2.Implement Logistic Regression using Stochastic Gradient Descent.

gradient_descent=function(X,Y,theta,alpha=0.3,epsilon=0.001,iter=100){
  i=0
  N=length(Y)
  repeat{
    i=i+1
    if(i%%10==0){
      print(paste0(i,"th iteration"))
    }
    #randomly pick 1 samples to calculate gradient for updating theta
    sampled=sample(1:N,1)
    #vectorized calculation for gradient
    gradient=Y[sampled]*X[,sampled]/(1+exp(sum(-Y[sampled]*theta*X[,sampled])))
    if(i>iter ){
      break
    } 
    else if(sqrt(sum(gradient^2))<epsilon){
        
        #break
    }
    if(i%%50==0){
      print(i)
      #print("break by norm")
      print(sqrt(sum(gradient^2)))
    }
    
    #update new theta
    theta=theta-alpha*gradient
    alpha=alpha*0.9995
  }
  #return theta
  theta
}

logistic_regression=function(X_train,Y_train,initial_theta=0,
                             alpha=0.3,epsilon=0.001,iter=100){
  #Get number of samples
  N=length(Y_train)
  #transform Y_train to -1 and 1
  twoclass=unique(Y_train)
  m=mean(twoclass)
  s=abs(twoclass[1]-twoclass[2])/2
  Y=(Y_train-m)/s
  
  #transform dataframe to matrix
  Xm=as.matrix(X_train)
  #Add bias term 1
  X=rbind(rep(1,N),Xm)
  #initial theta 
  theta=rep(initial_theta,nrow(X))
  # start gradient descent loop and make model
  
  model=gradient_descent(X,Y,theta,alpha,epsilon,iter)
  #retrun model
  model
}

##3. Training Stochasti Gradient Descent model

#3a. Train 2 models, one on the train_0_1 set and another on train_3_5, 
#and report the training and test accuracies

#make model to classify 0 and 1
model_0_1=logistic_regression(train_0_1,true_label_train_0_1,alpha=1,initial_theta = 1,iter = 100)

#make model to classify 3 and 5
model_3_5=logistic_regression(train_3_5,true_label_train_3_5,alpha=1,iter = 100)

#report in sample accuracy and out of sample accuray for classification of 0 and 1
train_0_1_accuracy=get_accuracy(train_0_1,true_label_train_0_1,model_0_1)
test_0_1_accuracy=get_accuracy(test_0_1,true_label_test_0_1,model_0_1)

print(paste("The accuracy for model_0_1 on train_0_1 is:",train_0_1_accuracy))
print(paste("The accuracy for model_0_1 on test_0_1 is:",test_0_1_accuracy))


#report in sample accuracy and out of sample accuray for classification of 3 and 5
train_3_5_accuracy=get_accuracy(train_3_5,true_label_train_3_5,model_3_5)
test_3_5_accuracy=get_accuracy(test_3_5,true_label_test_3_5,model_3_5)

print(paste("The accuracy for model_3_5 on train_3_5 is:",train_3_5_accuracy))
print(paste("The accuracy for model_3_5 on test_3_5 is:",test_3_5_accuracy))

#helper function to calculate accuracy 
get_accuracy=function(Xdf,Ydf,model){
  #Get number of samples
  N=length(Ydf)
  #transform Y_train to -1 and 1
  twoclass=unique(Ydf)
  m=mean(twoclass)
  s=abs(twoclass[1]-twoclass[2])/2
  Y=(Ydf-m)/s
  #transform dataframe to matrix
  Xm=as.matrix(Xdf)
  #Add bias term 1
  X=rbind(rep(1,N),Xm)
  #predict Y by model
  pY=predict_y(X,model)
  #evaluate accuracy
  sum(Y==pY)/length(Y)
}

predict_y=function(X,model){
  
  p1=1/as.vector(1+exp(model%*%X))
  predY=c()
  predY[p1>=0.5]=1
  predY[p1<0.5]=-1
  predY
}

#3b: train on 10 random 80% divisions of training data and report average accuracies

for(i in 1:10){
  N01=length(true_label_train_0_1)
  sampled01=sample(1:N01,floor(0.8*N01))
  model01=logistic_regression(train_0_1[,sampled01],
                              true_label_train_0_1[sampled01],
                              alpha=1,iter = 100)
  
  N35=length(true_label_train_3_5)
  sampled35=sample(1:N35,floor(0.8*N35))
  model35=logistic_regression(train_3_5[,sampled35],
                              true_label_train_3_5[sampled35],
                              alpha=1,iter = 100)

  if(i==1){
    accu_list01_train=c()
    accu_list01_test=c()
    accu_list35_train=c()
    accu_list35_test=c()
  }
  
  accu01train=get_accuracy(train_0_1[,sampled01],true_label_train_0_1[sampled01],model01)
  accu01test=get_accuracy(test_0_1,true_label_test_0_1,model01)
  
  accu35train=get_accuracy(train_3_5[,sampled35],true_label_train_3_5[sampled35],model35)
  accu35test=get_accuracy(test_3_5,true_label_test_3_5,model35)
  
  accu_list01_train=c(accu_list01_train,accu01train)
  accu_list01_test=c(accu_list01_test,accu01test)
  accu_list35_train=c(accu_list35_train,accu35train)
  accu_list35_test=c(accu_list35_test,accu35test)
  
}
print(paste("Average in sample accuracy for classification of 0 and 1 is",mean(accu_list01_train)))
print(paste("Average out of sample accuracy for classification of 0 and 1 is",mean(accu_list01_test)))
print(paste("Average in sample accuracy for classification of 3 and 5 is",mean(accu_list35_train)))
print(paste("Average out of sample accuracy for classification of 3 and 5 is",mean(accu_list35_test)))



#4a,4b: Apply your "new criteria" and report average accuracies on 10 random samples of 80% of training data.

#try small train sample to find best parameters for theta
averageAccuTrain35=c()
ini_thetas=seq(-1,1,length.out = 11)
for (ini_theta in ini_thetas){
  for(i in 1:10){
  
    N35=length(true_label_train_3_5)
    sampled35=sample(1:N35,floor(0.05*N35))
    model35_4a=logistic_regression(train_3_5[,sampled35],
                                   true_label_train_3_5[sampled35],
                                   initial_theta = ini_theta,
                                   alpha=0.5,iter = 100)
    
    if(i==1){
      accu_list35_train_4a=c()
      #accu_list35_test_4a=c()
    }
    
    
    accu35train=get_accuracy(train_3_5[,sampled35],true_label_train_3_5[sampled35],model35_4a)
    #accu35test=get_accuracy(test_3_5,true_label_test_3_5,model35_4a)
    
    accu_list35_train_4a=c(accu_list35_train_4a,accu35train)
    #accu_list35_test_4a=c(accu_list35_test_4a,accu35test)

  }
  averageAccuTrain35=c(averageAccuTrain35,mean(accu_list35_train_4a))
}

plot(inithetas,averageAccuTrain35)


#set all theta initialized at 0.3 
for(i in 1:10){

  N35=length(true_label_train_3_5)
  sampled35=sample(1:N35,floor(0.8*N35))
  model35_4a=logistic_regression(train_3_5[,sampled35],
                              true_label_train_3_5[sampled35],
                              initial_theta = 0.3,
                              alpha=1,iter = 100)
  
  if(i==1){
    accu_list35_train_4a=c()
    accu_list35_test_4a=c()
  }
  

  accu35train=get_accuracy(train_3_5[,sampled35],true_label_train_3_5[sampled35],model35_4a)
  accu35test=get_accuracy(test_3_5,true_label_test_3_5,model35_4a)
  
  accu_list35_train_4a=c(accu_list35_train_4a,accu35train)
  accu_list35_test_4a=c(accu_list35_test_4a,accu35test)
  
}
print(paste("4a, average in sample accuracy for classification of 3 and 5 is",mean(accu_list35_train_4a)))
print(paste("4a, average out of sample accuracy for classification of 3 and 5 is",mean(accu_list35_test_4a)))



averageAccuTrain35_4b=c()
epsilons=0.1^rep(1:5)
for (epsilon in epsilons){
  for(i in 1:10){
    
    N35=length(true_label_train_3_5)
    sampled35=sample(1:N35,floor(0.05*N35))
    model35_4b=logistic_regression(train_3_5[,sampled35],
                                   true_label_train_3_5[sampled35],
                                   initial_theta = 0,
                                   epsilon = epsilon,
                                   alpha=0.5,iter = 200)
    
    if(i==1){
      accu_list35_train_4b=c()
      #accu_list35_test_4a=c()
    }
    
    
    accu35train=get_accuracy(train_3_5[,sampled35],true_label_train_3_5[sampled35],model35_4b)
    #accu35test=get_accuracy(test_3_5,true_label_test_3_5,model35_4a)
    
    accu_list35_train_4b=c(accu_list35_train_4b,accu35train)
    #accu_list35_test_4a=c(accu_list35_test_4a,accu35test)
    
  }
  averageAccuTrain35_4b=c(averageAccuTrain35_4b,mean(accu_list35_train_4b))
}

plot(as.factor(epsilons),averageAccuTrain35_4b)






for(i in 1:10){
  
  N35=length(true_label_train_3_5)
  sampled35=sample(1:N35,floor(0.8*N35))
  model35_4b=logistic_regression(train_3_5[,sampled35],
                                 true_label_train_3_5[sampled35],
                                 initial_theta = 0,epsilon = 0.0001,
                                 alpha=0.5,iter = 200)
  
  if(i==1){
    accu_list35_train_4b=c()
    accu_list35_test_4b=c()
  }
  
  
  accu35train=get_accuracy(train_3_5[,sampled35],true_label_train_3_5[sampled35],model35_4b)
  accu35test=get_accuracy(test_3_5,true_label_test_3_5,model35_4b)
  
  accu_list35_train_4b=c(accu_list35_train_4b,accu35train)
  accu_list35_test_4b=c(accu_list35_test_4b,accu35test)
  
}
print(paste("4b, average in sample accuracy for classification of 3 and 5 is",mean(accu_list35_train_4b)))
print(paste("4b, average out of sample accuracy for classification of 3 and 5 is",mean(accu_list35_test_4b)))


#5

averageAccuTrain10_5a=c()
averageAccuTest10_5a=c()
averageAccuTrain35_5a=c()
averageAccuTest35_5a=c()
averageCostTrain10_5b=c()
averageCostTest10_5b=c()
averageCostTrain35_5b=c()
averageCostTest35_5b=c()
train_size=c(1:20)*0.05
for (size in train_size){
  for(i in 1:10){
    
    N01=length(true_label_train_0_1)
    sampled01=sample(1:N01,floor(size*N01))
    model01_5a=logistic_regression(train_0_1[,sampled01],
                                true_label_train_0_1[sampled01],
                                initial_theta = 0,
                                alpha=0.5,iter = 200)
    
    N35=length(true_label_train_3_5)
    sampled35=sample(1:N35,floor(size*N35))
    model35_5a=logistic_regression(train_3_5[,sampled35],
                                   true_label_train_3_5[sampled35],
                                   initial_theta = 0,
                                   alpha=0.5,iter = 200)
    
    if(i==1){
      accu_list10_train_5a=c()
      accu_list10_test_5a=c()
      accu_list35_train_5a=c()
      accu_list35_test_5a=c()
      
      cost_list10_train_5b=c()
      cost_list10_test_5b=c()
      cost_list35_train_5b=c()
      cost_list35_test_5b=c()
    }
    accu01train=get_accuracy(train_0_1[,sampled01],true_label_train_0_1[sampled01],model01_5a)
    accu01test=get_accuracy(test_0_1,true_label_test_0_1,model01_5a)
    
    accu_list10_train_5a=c(accu_list10_train_5a,accu01train)
    accu_list10_test_5a=c(accu_list10_test_5a,accu01test)
    
    accu35train=get_accuracy(train_3_5[,sampled35],true_label_train_3_5[sampled35],model35_5a)
    accu35test=get_accuracy(test_3_5,true_label_test_3_5,model35_5a)
    
    accu_list35_train_5a=c(accu_list35_train_5a,accu35train)
    accu_list35_test_5a=c(accu_list35_test_5a,accu35test)
    
    
    cost10train=cost(train_0_1[,sampled01],true_label_train_0_1[sampled01],model01_5a)
    cost10test=cost(test_0_1,true_label_test_0_1,model01_5a)
    
    cost_list10_train_5b=c(cost_list10_train_5b,cost10train)
    cost_list10_test_5b=c(cost_list10_test_5b,cost10test)
    
    cost35train=cost(train_3_5[,sampled35],true_label_train_3_5[sampled35],model35_5a)
    cost35test=cost(test_3_5,true_label_test_3_5,model35_5a)    
    
    cost_list35_train_5b=c(cost_list35_train_5b,cost35train)
    cost_list35_test_5b=c(cost_list35_test_5b,cost35test)
  }
  averageAccuTrain10_5a=c(averageAccuTrain10_5a,mean(accu_list10_train_5a))
  averageAccuTest10_5a=c(averageAccuTest10_5a,mean(accu_list10_test_5a))
  averageAccuTrain35_5a=c(averageAccuTrain35_5a,mean(accu_list35_train_5a))
  averageAccuTest35_5a=c(averageAccuTest35_5a,mean(accu_list35_test_5a))
  
  averageCostTrain10_5b=c(averageCostTrain10_5b,mean(cost_list10_train_5b))
  averageCostTest10_5b=c(averageCostTest10_5b,mean(cost_list10_test_5b))
  averageCostTrain35_5b=c(averageCostTrain35_5b,mean(cost_list35_train_5b))
  averageCostTest35_5b=c(averageCostTest35_5b,mean(cost_list35_test_5b))
}

plot(as.factor(train_size),averageAccuTrain10_5a)



cost=function(Xdf,Ydf,model){
  #Get number of samples
  N=length(Ydf)
  #transform Y_train to -1 and 1
  twoclass=unique(Ydf)
  m=mean(twoclass)
  s=abs(twoclass[1]-twoclass[2])/2
  Y=(Ydf-m)/s
  #transform dataframe to matrix
  Xm=as.matrix(Xdf)
  #Add bias term 1
  X=rbind(rep(1,N),Xm)
  #calculate cost
  mean(log(1+exp(Y*model%*%X)))
}

error_rate=function(Y,predY){
  sum(Y!=predY)/length(Y)
}


