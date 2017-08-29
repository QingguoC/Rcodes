
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
#rotate image
str0=t(str0[28:1,])
trueLabTr0=true_label_train_0_1[selectTr0]
image(str0,col=gray.colors(256), xlab = trueLabTr0,main="Class 0 from train_0_1")

#sample one column for class 1 to plot in test_0_1
selectTe1=sample(which(true_label_test_0_1==1),1)
ste1=matrix(test_0_1[,selectTe1],ncol = 28)
#rotate image
ste1=t(ste1[28:1,])
trueLabTe1=true_label_test_0_1[selectTe1]
image(ste1,col=gray.colors(256),xlab = trueLabTe1,main="Class 1 from test_0_1")

#sample one column for class 3 to plot in train_3_5
selectTr3=sample(which(true_label_train_3_5==3),1)
str3=matrix(train_3_5[,selectTr3],ncol = 28)
#rotate image
str3=t(str3[28:1,])
trueLabTr3=true_label_train_3_5[selectTr3]
image(str3,col=gray.colors(256),xlab = trueLabTr3,main="Class 3 from train_3_5")

#sample one column for class 5 to plot in test_3_5
selectTe5=sample(which(true_label_test_3_5==5),1)
ste5=matrix(test_3_5[,selectTe5],ncol = 28)
#rotate image
ste5=t(ste5[28:1,])
trueLabTe5=true_label_test_3_5[selectTe5]
image(ste5,col=gray.colors(256),xlab = trueLabTe5,main="Class 5 from test_3_5")

par(mfrow=c(1,1)) #reset par setting

##Problem 2.Implement Logistic Regression using Batch Gradient Descent.

#gradient_descent is a function with iterations 
#to update model in logistic regression by calculating gradients

#X is matrix to train with rows of features and a row of bias, and columns of samples
#Y is a vector of transformed true labels with -1,1 for all train samples
#alpha is learning rate
#epsilon is convergence threshold
#maxiter is max limit of iterations
#miniter is min limit of iterations to avoid unexpected early stop
#converge_criteria is the criteria used to stop unnecessary iterations
gradient_descent=function(X,Y,theta,alpha,epsilon,maxiter,miniter,converge_criteria){
  i=0
  N=length(Y)#number of samples
  D=nrow(X)
  oldloss=mean(log(1+exp(Y*theta%*%X))) #calculate initial loss value
  #start iterations
  repeat{
    i=i+1
    #vectorized calculation for gradient
    #This is the core part of the codes
    #Leave X alone as numerator to make computing faster
    gradient=rowSums(X/matrix(Y*(1+exp(-Y*theta%*%X)),nrow = D,ncol = N,T))/N
    
    if(i>maxiter ){
      #when exceed maximum number of iterations, break iteration
      break
    } 
    else if(i>miniter){
      #Default convergence criteria by checking changes of gradient using norm function 
      if(converge_criteria=="F"){
        if(abs(norm(as.matrix(oldGradient),"F")-norm(as.matrix(gradient),"F"))<epsilon){
          print("break by norm")
          break
        }
      }
      #criteria used for 4b by checking difference of losses by iterations
      else if(converge_criteria=="L"){
        loss=mean(log(1+exp(Y*theta%*%X))) # this is loss function I used in this project
        if(abs(oldloss-loss)<epsilon){
          print("break by Loss")
          break
        }
        oldloss=loss # record this loss value for next comparison
      }
      
    }
    #update new theta
    theta=theta-alpha*gradient
    alpha=alpha*0.99 #let learning rate decay by iterations
    oldGradient=gradient #record this gradient for next comparison
  }
  print(paste0(i,"th iteration stops"))
  #return theta
  theta
}

#main function for logistic regression
#X_train is dataframe to train with rows of features and columns of samples
#Y_train is a vector of true labels for all train samples
#alpha is learning rate
#epsilon is convergence threshold
#maxiter is max limit of iterations
#miniter is min limit of iterations to avoid unexpected early stop
#converge_criteria is the criteria used to stop unnecessary iterations
logistic_regression=function(X_train,Y_train,initial_theta=0,
                             alpha=1,epsilon=0.001,
                             maxiter=500,miniter=10,
                             converge_criteria="F"){
  #Get number of samples
  N=length(Y_train)
  #transform Y_train to -1 and 1
  twoclass=unique(Y_train)#extract two classes in Y_train
  m=mean(twoclass)#calculate the mean
  s=abs(twoclass[1]-twoclass[2])/2 #calculate the half range
  Y=(Y_train-m)/s #transform Y_train to -1 and 1
  
  #transform dataframe to matrix
  Xm=as.matrix(X_train)
  #Add bias term 1
  X=rbind(rep(1,N),Xm)
  #initial theta 
  theta=rep(initial_theta,nrow(X))
  # start gradient descent loop and make model
  
  model=gradient_descent(X,Y,theta,alpha,epsilon,maxiter,miniter,converge_criteria)
  #retrun model
  model
}

##3. Training Batch Gradient Descent model

#3a. Train 2 models, one on the train_0_1 set and another on train_3_5, 
#and report the training and test accuracies

#make model to classify 0 and 1
model_0_1=logistic_regression(train_0_1,true_label_train_0_1)

#make model to classify 3 and 5
model_3_5=logistic_regression(train_3_5,true_label_train_3_5)


#helper function to calculate accuracy 
get_accuracy=function(Xdf,Y,model){
  #Get number of samples
  N=length(Y)
  #get two classes from Y_train 
  twoclass=sort(unique(Y))
  classNeg1=twoclass[1]
  classPos1=twoclass[2]
  #transform dataframe to matrix
  Xm=as.matrix(Xdf)
  #Add bias term 1
  X=rbind(rep(1,N),Xm)
  #predict Y by model
  pY=predict_y(X,model,classNeg1,classPos1)
  #evaluate accuracy
  sum(Y==pY)/length(Y)
}
#helper function to predict Y, 
#classNeg1 is the label treated as -1 in the model
#classPos1 is the label treated as 1 in the model
predict_y=function(X,model,classNeg1,classPos1){
  #calculate probability
  p1=1/as.vector(1+exp(model%*%X))

  predY=c()
  #assign predicted labels
  predY[p1>=0.5]=classPos1
  predY[p1<0.5]=classNeg1
  predY
}

#report in sample accuracy and out of sample accuray for classification of 0 and 1
train_0_1_accuracy=get_accuracy(train_0_1,true_label_train_0_1,model_0_1)
test_0_1_accuracy=get_accuracy(test_0_1,true_label_test_0_1,model_0_1)

print(paste("3a. The accuracy of model_0_1 on train_0_1 is:",train_0_1_accuracy))
print(paste("3a. The accuracy of model_0_1 on test_0_1 is:",test_0_1_accuracy))


#report in sample accuracy and out of sample accuray for classification of 3 and 5
train_3_5_accuracy=get_accuracy(train_3_5,true_label_train_3_5,model_3_5)
test_3_5_accuracy=get_accuracy(test_3_5,true_label_test_3_5,model_3_5)

print(paste("The accuracy of model_3_5 on train_3_5 is:",train_3_5_accuracy))
print(paste("The accuracy of model_3_5 on test_3_5 is:",test_3_5_accuracy))



#3b: train on 10 random 80% divisions of training data and report average accuracies
N01=length(true_label_train_0_1)
N35=length(true_label_train_3_5)
#run 10 times to build models and evaluations for each classification
for(i in 1:10){
  #randomly sample 80% train data for class 0,1
  sampled01=sample(1:N01,floor(0.8*N01))
  #train model for class 0,1
  model01=logistic_regression(train_0_1[,sampled01],
                              true_label_train_0_1[sampled01])
  
  #randomly sample 80% train data for class 3,5
  sampled35=sample(1:N35,floor(0.8*N35))
  #train model for class 3,5
  model35=logistic_regression(train_3_5[,sampled35],
                              true_label_train_3_5[sampled35])

  if(i==1){
    #define a set of vectors to store accuracies accordingly
    accu_list01_train=c()
    accu_list01_test=c()
    accu_list35_train=c()
    accu_list35_test=c()
  }
  
  #use helper function get_accuracy to calculate accuracy
  accu01train=get_accuracy(train_0_1[,sampled01],true_label_train_0_1[sampled01],model01)
  accu01test=get_accuracy(test_0_1,true_label_test_0_1,model01)
  
  accu35train=get_accuracy(train_3_5[,sampled35],true_label_train_3_5[sampled35],model35)
  accu35test=get_accuracy(test_3_5,true_label_test_3_5,model35)
  
  #add the above results to the vectors respectively
  accu_list01_train=c(accu_list01_train,accu01train)
  accu_list01_test=c(accu_list01_test,accu01test)
  accu_list35_train=c(accu_list35_train,accu35train)
  accu_list35_test=c(accu_list35_test,accu35test)
  
}
print(paste("3b. Average accuracy for classification of 0 and 1 in train is",mean(accu_list01_train)))
print(paste("3b. Average accuracy for classification of 0 and 1 in test is",mean(accu_list01_test)))
print(paste("3b. Average accuracy for classification of 3 and 5 in train is",mean(accu_list35_train)))
print(paste("3b. Average accuracy for classification of 3 and 5 in test is",mean(accu_list35_test)))

#create dataframe df_3b for later plot
accuracy3b=c(mean(accu_list01_train),mean(accu_list01_test),mean(accu_list35_train),mean(accu_list35_test))
evaluation_sample=c("train_0_1_3b","test_0_1_3b","train_3_5_3b","test_3_5_3b")
df_3b=data.frame(evaluation_sample=evaluation_sample,
                 accuracy=accuracy3b)


library(ggplot2)
#plot results of 3b
ggplot(data=df_3b,aes(evaluation_sample,accuracy))+
  geom_bar(stat="identity",aes(fill=evaluation_sample))+
  coord_cartesian(ylim = c(0.9, 1))+
  scale_x_discrete(limit=evaluation_sample)+
  ylab("Average accuracy")+
  geom_text(aes(label = round(accuracy,digits = 6),y=accuracy+0.004), size = 4)+
  ggtitle("3b Evaluation of models made from 80% train data")

#4a,4b: Apply your "new criteria" and report average accuracies on 10 random samples of 80% of training data.

#to try other initial_theta=0.5 
#to compare two initial thetas, 0 and 0.5, I tuned alpha from 0.3 to 300 for both.
#try to find best alpha for each.
start4a=Sys.time() #record start time
#create a set of vectors to record the experiment
Accu35_4a=c()
alphas=c()
thetas=c()
types=c()
N35=length(true_label_train_3_5)
for(ini_theta in c(0,0.5)){ #two candidates for initial theta
  for(ini_alpha in c(0.3,1,3,10,30,100,300)){ #seven candidates of initial alpha
    for(i in 1:10){#repeat 10 times for averaging
  
      sampled35=sample(1:N35,floor(0.8*N35))#randomly sample 80% train data
      #build a model with ini_alpha and ini_theta
      model35_4a=logistic_regression(train_3_5[,sampled35],
                                     true_label_train_3_5[sampled35],
                                     alpha = ini_alpha,
                                     initial_theta =ini_theta)
      
      #get train and test accuracies
      accu35train=get_accuracy(train_3_5[,sampled35],true_label_train_3_5[sampled35],model35_4a)
      accu35test=get_accuracy(test_3_5,true_label_test_3_5,model35_4a)
      #append records
      Accu35_4a=c(Accu35_4a,accu35train,accu35test)
      alphas=c(alphas,ini_alpha,ini_alpha)
      thetas=c(thetas,ini_theta,ini_theta)
      types=c(types,"train_3_5","test_3_5")
    }
    #visualize the progress
    print(paste("Ends at theta",ini_theta," and alpha",ini_alpha))
  }
}

end4a=Sys.time()
#record the end time of iterations and show runtime
print(end4a-start4a)

#make a dataframe for later plot
df_4a_0=data.frame( accuracy=Accu35_4a,
                    theta=as.factor(thetas),
                    alpha=as.factor(alphas),
                    type=as.factor(types)
)

#summarize average and standard error of the accuracies.
df4a_acc=aggregate(df_4a_0["accuracy"],by=df_4a_0[c("theta","alpha","type")],function(X) c(mean=mean(X),sd=sd(X),N=length(X)))
df4a_acc=do.call(data.frame,df4a_acc)
names(df4a_acc)[4:6]=c("mean","sd","N")
df4a_acc$se=df4a_acc$sd/sqrt(df4a_acc$N)#calculate se

#make the plot
ggplot(df4a_acc,aes(alpha,mean,group=theta,col=theta))+
  geom_point()+
  geom_line()+
  facet_grid(.~type)+
  geom_errorbar(aes(ymin=mean-se,ymax=mean+se),width=0.2)+
  ylab("Average accuracy")+
  ggtitle("4a models with different initializations compared to 3b")

#sort df4a_acc by type, theta and decreasing order of accuracy
sorted_df4a_acc=df4a_acc[order(df4a_acc$type,df4a_acc$theta,-df4a_acc$mean),]

#extract best train accuracy when intial theta=0
sorted_df4a_acc[sorted_df4a_acc$theta==0 &sorted_df4a_acc$type=="train_3_5" ,][1,]
#extract best test accuracy when intial theta=0
sorted_df4a_acc[sorted_df4a_acc$theta==0 &sorted_df4a_acc$type=="test_3_5" ,][1,]
#extract best train accuracy when intial theta=0.5
sorted_df4a_acc[sorted_df4a_acc$theta==0.5 &sorted_df4a_acc$type=="train_3_5" ,][1,]
#extract best test accuracy when intial theta=0.5
sorted_df4a_acc[sorted_df4a_acc$theta==0.5 &sorted_df4a_acc$type=="test_3_5" ,][1,]

#4b Experiment with different convergence criteria for gradient descent
#previously we calculate the difference of Frobenius norm of old and new gradient,
#and compare it with epsilon

#Here I try a new criteria
#Since we optimize models by minimizing loss, we can directly calculate loss
#and calculate the difference of loss values from the current and last iterations,
#and compare it with epsilon

#the new converge_criteria code is in the same gradient_descent function
# we only need call it with parameter converge_criteria = "L"

#a set of vectors to record experiment
Accu35_4b_train=c()
Accu35_4b_test=c()
N35=length(true_label_train_3_5)
for(i in 1:10){
  
  #randomly sample 80% train data
  sampled35=sample(1:N35,floor(0.8*N35))
  #build a model with new converge_criteria "L"
  model35_4b=logistic_regression(train_3_5[,sampled35],
                                 true_label_train_3_5[sampled35],
                                converge_criteria = "L")
  
  #evaluate model in train and test sample
  accu35train=get_accuracy(train_3_5[,sampled35],true_label_train_3_5[sampled35],model35_4b)
  accu35test=get_accuracy(test_3_5,true_label_test_3_5,model35_4b)
  
  #append records
  Accu35_4b_train=c(Accu35_4b_train,accu35train)
  Accu35_4b_test=c(Accu35_4b_test,accu35test)
}

#print average accuracies
print(paste("4bAverage accuracy for classification of 3 and 5 in train is",mean(Accu35_4b_train)))
print(paste("4bAverage accuracy for classification of 3 and 5 in test is",mean(Accu35_4b_test)))

#make a dataframe df_4b for plot
accuracy4b=c(mean(Accu35_4b_train),mean(Accu35_4b_test))
evaluation_sample=c("train_3_5_4b","test_3_5_4b")
df_4b1=data.frame(accuracy=accuracy4b,
                  evaluation_sample=evaluation_sample)

df_4b=rbind(df_3b[3:4,],df_4b1)

#plot the results of 4b with 3b 
ggplot(data=df_4b,aes(evaluation_sample,accuracy))+
  geom_bar(stat="identity",aes(fill=evaluation_sample))+
  coord_cartesian(ylim = c(0.8, 1))+
  scale_x_discrete(limit=df_4b$evaluation_sample)+
  geom_text(aes(label = round(accuracy,digits = 6),y=accuracy+0.0075), size = 4)+
  ggtitle("4b with converge_criteria by checking loss change\ncompared to 3b with converge_criteria\nby checking gradient change ")

#5 Learning Curves
#Here I run experiments for 5a and 5b together

#first make a helper function to calculate sum negative log likelihood
sumNegLogLikelihood=function(Xdf,Ydf,model){
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
  #calculate sum negative log likelihood
  sum(log(1+exp(Y*model%*%X)))
}
#Then make a helper function to calculate mean negative log likelihood
meanNegLogLikelihood=function(Xdf,Ydf,model){
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
  #calculate sum negative log likelihood
  mean(log(1+exp(Y*model%*%X)))
}

#start the experiment
start=Sys.time() #record start time
#create a set of vectors to record experiment
accuracy5a=c() 
sumNegLog5b=c()
meanNegLog5b=c()
class5=c()
sizes=c()

train_size=c(1:20)*0.05 #20 train sizes
N01=length(true_label_train_0_1)
N35=length(true_label_train_3_5)

for (size in train_size){ #iterate each train size
  for(i in 1:10){ #repeat 10 times
    
    #randomly sample certain size of data from train 0,1
    sampled01=sample(1:N01,floor(size*N01))
    #build model from the sampled data
    model01_5a=logistic_regression(train_0_1[,sampled01],
                                   true_label_train_0_1[sampled01])
    
    #randomly sample certain size of data from train 3,5
    sampled35=sample(1:N35,floor(size*N35))
    #build model from the sampled data
    model35_5a=logistic_regression(train_3_5[,sampled35],
                                   true_label_train_3_5[sampled35])
    
    #evaluate model for class 0,1
    accu01train=get_accuracy(train_0_1[,sampled01],true_label_train_0_1[sampled01],model01_5a)
    accu01test=get_accuracy(test_0_1,true_label_test_0_1,model01_5a)
    
    #evaluate model for class 3,5
    accu35train=get_accuracy(train_3_5[,sampled35],true_label_train_3_5[sampled35],model35_5a)
    accu35test=get_accuracy(test_3_5,true_label_test_3_5,model35_5a)
    
    #get sum negative log likelihood for class 0,1
    cost10train=sumNegLogLikelihood(train_0_1[,sampled01],true_label_train_0_1[sampled01],model01_5a)
    cost10test=sumNegLogLikelihood(test_0_1,true_label_test_0_1,model01_5a)
    #get sum negative log likelihood for class 3,5
    cost35train=sumNegLogLikelihood(train_3_5[,sampled35],true_label_train_3_5[sampled35],model35_5a)
    cost35test=sumNegLogLikelihood(test_3_5,true_label_test_3_5,model35_5a)    

    #get mean negative log likelihood for class 0,1
    meancost10train=meanNegLogLikelihood(train_0_1[,sampled01],true_label_train_0_1[sampled01],model01_5a)
    meancost10test=meanNegLogLikelihood(test_0_1,true_label_test_0_1,model01_5a)
    #get mean negative log likelihood for class 3,5
    meancost35train=meanNegLogLikelihood(train_3_5[,sampled35],true_label_train_3_5[sampled35],model35_5a)
    meancost35test=meanNegLogLikelihood(test_3_5,true_label_test_3_5,model35_5a) 
    
    #append new records
    accuracy5a=c(accuracy5a,accu01train,accu01test,accu35train,accu35test)
    sumNegLog5b=c(sumNegLog5b,cost10train,cost10test,cost35train,cost35test)
    meanNegLog5b=c(meanNegLog5b,meancost10train,meancost10test,meancost35train,meancost35test)
    class5=c(class5,"train_0_1","test_0_1","train_3_5","test_3_5")
    sizes=c(sizes,rep(size,4))
  }
  #show the progress
  print(paste("progress to",size))
}

end=Sys.time()#the end time

print(end-start)#print runtime

#create a dataframe for further statistical analysis
df5ab=data.frame(size=as.factor(sizes),
                 accuracy=accuracy5a,
                 sumNegLog=sumNegLog5b,
                 meanNegLog=meanNegLog5b,
                 evaluation_sample=class5
)
#summarize average and se of accuracies
df5_acc=aggregate(df5ab["accuracy"],by=df5ab[c("size","evaluation_sample")],function(X) c(mean=mean(X),sd=sd(X),N=length(X)))
df5_acc=do.call(data.frame,df5_acc)
names(df5_acc)[3:5]=c("mean","sd","N")
df5_acc$se=df5_acc$sd/sqrt(df5_acc$N)
#seperate class 0,1 and class 3,5 from above dataframe to two dataframes for plot
df5_acc_01=subset(df5_acc,evaluation_sample=="train_0_1" | evaluation_sample=="test_0_1")
df5_acc_35=subset(df5_acc,evaluation_sample=="train_3_5" | evaluation_sample=="test_3_5")

#plot accuracy plot of 5a for class 0,1
ggplot(df5_acc_01,aes(size,mean,group=evaluation_sample,col=evaluation_sample))+
  geom_point()+
  geom_line()+
  geom_errorbar(aes(ymin=mean-se,ymax=mean+se),width=0.2)+
  ylab("accuracy")+
  theme(axis.text.x = element_text(angle=90, hjust=1))+
  ggtitle("Average accuracy by training size for Class 0 and 1")

#plot accuracy plot of 5a for class 3,5
ggplot(df5_acc_35,aes(size,mean,group=evaluation_sample,col=evaluation_sample))+
  geom_point()+
  geom_line()+
  geom_errorbar(aes(ymin=mean-se,ymax=mean+se),width=0.2)+
  ylab("accuracy")+
  theme(axis.text.x = element_text(angle=90, hjust=1))+
  ggtitle("Average accuracy by training size for Class 3 and 5")

#summarize average and se of sum negative log likelihood 
df5_sumNegLog=aggregate(df5ab["sumNegLog"],by=df5ab[c("size","evaluation_sample")],function(X) c(mean=mean(X),sd=sd(X),N=length(X)))
df5_sumNegLog=do.call(data.frame,df5_sumNegLog)
names(df5_sumNegLog)[3:5]=c("mean","sd","N")
df5_sumNegLog$se=df5_sumNegLog$sd/sqrt(df5_sumNegLog$N)

#seperate class 0,1 and class 3,5 from above dataframe to two dataframes for plot
df5_sumNegLog_01=subset(df5_sumNegLog,evaluation_sample=="train_0_1" | evaluation_sample=="test_0_1")
df5_sumNegLog_35=subset(df5_sumNegLog,evaluation_sample=="train_3_5" | evaluation_sample=="test_3_5")

#plot sum negative log likelihood of 5b for class 0,1
ggplot(df5_sumNegLog_01,aes(size,mean,group=evaluation_sample,col=evaluation_sample))+
  geom_point()+
  geom_line()+
  geom_errorbar(aes(ymin=mean-se,ymax=mean+se),width=0.2)+
  ylab("Sum_Negative_Log_Likelihood")+
  theme(axis.text.x = element_text(angle=90, hjust=1))+
  ggtitle("Average Sum_Negative_Log_Likelihood\n by training size for Class 0 and 1")

#plot sum negative log likelihood of 5b for class 3,5
ggplot(df5_sumNegLog_35,aes(size,mean,group=evaluation_sample,col=evaluation_sample))+
  geom_point()+
  geom_line()+
  geom_errorbar(aes(ymin=mean-se,ymax=mean+se),width=0.2)+
  ylab("Sum_Negative_Log_Likelihood")+
  theme(axis.text.x = element_text(angle=90, hjust=1))+
  ggtitle("Average Sum_Negative_Log_Likelihood\n by training size for Class 3 and 5")


#summarize average and se of mean negative log likelihood 
df5_meanNegLog=aggregate(df5ab["meanNegLog"],by=df5ab[c("size","evaluation_sample")],function(X) c(mean=mean(X),sd=sd(X),N=length(X)))
df5_meanNegLog=do.call(data.frame,df5_meanNegLog)
names(df5_meanNegLog)[3:5]=c("mean","sd","N")
df5_meanNegLog$se=df5_meanNegLog$sd/sqrt(df5_meanNegLog$N)

#seperate class 0,1 and class 3,5 from above dataframe to two dataframes for plot
df5_meanNegLog_01=subset(df5_meanNegLog,evaluation_sample=="train_0_1" | evaluation_sample=="test_0_1")
df5_meanNegLog_35=subset(df5_meanNegLog,evaluation_sample=="train_3_5" | evaluation_sample=="test_3_5")

#plot mean negative log likelihood of 5b for class 0,1
ggplot(df5_meanNegLog_01,aes(size,mean,group=evaluation_sample,col=evaluation_sample))+
  geom_point()+
  geom_line()+
  geom_errorbar(aes(ymin=mean-se,ymax=mean+se),width=0.2)+
  ylab("Mean_Negative_Log_Likelihood")+
  theme(axis.text.x = element_text(angle=90, hjust=1))+
  ggtitle("Average Mean_Negative_Log_Likelihood\n by training size for Class 0 and 1")

#plot mean negative log likelihood of 5b for class 3,5
ggplot(df5_meanNegLog_35,aes(size,mean,group=evaluation_sample,col=evaluation_sample))+
  geom_point()+
  geom_line()+
  geom_errorbar(aes(ymin=mean-se,ymax=mean+se),width=0.2)+
  ylab("Mean_Negative_Log_Likelihood")+
  theme(axis.text.x = element_text(angle=90, hjust=1))+
  ggtitle("Average Mean_Negative_Log_Likelihood\n by training size for Class 3 and 5")
