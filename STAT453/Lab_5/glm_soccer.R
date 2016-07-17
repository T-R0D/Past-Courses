#===================================#
#           STAT 453/653            #
#     France Soccer Championship    #
#         Season 2007-2008          #
#===================================#

# read data
Data<-read.table(file='France_2007_2008.txt',header=TRUE) 
T<-Data
W<-Data

# make R recognize the names of variables from "T"
attach(T) 

# Tie indicator
T<-factor(R1==1) # Tie indicator

#P1 Win indicator
W<-factor(R1==3)


# yellow background
par(bg='yellow')

# plot win indicator vs. player 1 points
windows()
plot(P1,jitter(as.numeric(W)),pch=19,col='blue')

# plot win indicator vs. P1-P2
windows()
plot(P1-P2,jitter(as.numeric(W),amount=.1),pch=19,col='blue',
ylab='Jittered Win Indicator (FALSE=1)')

# Histograms of point difference for tie/no tie
windows()
hist(P1[W==FALSE]-P2[T==FALSE],col='blue',
main='No tie',xlab='Point difference',xlim=c(-40,40))

windows()
hist(P1[W==TRUE]-P2[T==TRUE],col='blue',
main='Tie',xlab='Point difference',xlim=c(-40,40))

# GLMs for ties
#===============================================
model1<-glm(T~I(P1-P2)+I((P1-P2)^2),family=binomial)
model2<-glm(T~I((P1-P2)^2),family=binomial)

summary(model1)
summary(model2)

model3<-glm(T~I(P1-P2),family=binomial) # No good
summary(model3)

model4<-glm(T~I(P1*P2),family=binomial) # No good
summary(model4)

model5<-glm(T~I((P1-P2)^4),family=binomial) # sort of good
summary(model5)                             #============

model6<-glm(T~I((P1-P2)^3),family=binomial) # No good
summary(model6)

model7<-glm(T~I(P1-P2)+I((P1-P2)^2),family=binomial) # No good
summary(model7)

model8<-glm(T~I((P1-P2)^2)+I(P1*P2),family=binomial) # No good
summary(model8)

model9<-glm(T~I((P1-P2)^2)+I((P1-P2)^4),family=binomial) # No good
summary(model9)

model10<-glm(T~I(P1-P2)+I((P1-P2)^2)+I((P1-P2)^4),family=binomial) # No good
summary(model10)

model11<-glm(T~I(P1-P2)+I((P1-P2)^2)+I((P1-P2)^3)+I((P1-P2)^4),family=binomial) # No good
summary(model11)

model12<-glm(T~P1+I((P1-P2)^4),family=binomial) # No good
summary(model12)

# Candidates for best model
#===========================================================
summary(model1)
summary(model2)
summary(model5)


# GLMs for wins
#===============================================
win1<-glm(W~I((P1-P2)^2),family=binomial)
win2<-glm(W~I(P1-P2)+I((P1-P2)^2),family=binomial)
win3<-glm(W~I(P1-P2)+I((P1-P2)^4),family=binomial)
win4<-glm(W~I(P1-P2)+I(P1*P2),family=binomial)
win5<-glm(W~I(P1-P2)+I((P1-P2)^3),family=binomial)
win6<-glm(W~I((P1-P2)^3),family=binomial)              #best model
win7<-glm(W~I((P1-P2)^2)+I((P1-P2)^3),family=binomial)
win8<-glm(W~I(P1-P2)+I((P1-P2)^2)+I((P1-P2)^3),family=binomial)

summary(win1)
summary(win2)
summary(win3)
summary(win4)
summary(win5)
summary(win6)
summary(win7)
summary(win8)



# CI for model parameters
#===============================================
confint(win6,level=.95)
confint(model2,level=.99)

# CI for predicted proportions
#===============================================================
p<-predict(win6,data.frame(P1=50,P2=10),type='response',se.fit=TRUE)
L<-p$fit - qnorm(1-.025)*p$se.fit
R<-p$fit + qnorm(1-.025)*p$se.fit

# Plot predictions
#=============================================================

plot(P1,T)
plot(P1,predict(model1,data.frame(P1),type='response'),
     pch=19,col='blue',ylab='Probability of a tie')
grid()

windows()
plot(P1-P2,predict(win6,data.frame(P1,P2),type='response'),
pch=19,col='blue',xlab = 'Difference in Score (With Respect To P1', ylab='Probability of P1 Winning')
grid()

#windows()
plot(P1-P2,predict(model2,data.frame(P1,P2),type='response'),
     pch=19,col='blue',ylab='Probability of a tie')
grid()

windows()
plot(P1-P2,predict(model5,data.frame(P1,P2),type='response'),
     pch=19,col='blue',ylab='Probability of a tie')
grid()


# exact Numerical predictions
#================================================
predict(model5,data.frame(50,10),type='response')