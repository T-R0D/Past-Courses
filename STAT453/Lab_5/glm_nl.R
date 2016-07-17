#===================================#
#           STAT 453/653            #
#        "Non-linear" GLMs          #
#===================================#

T<-read.table(file='ttt.txt',header=TRUE) # reads data
attach(T) # makes R recognize the names of variables from "T"

par(bg='yellow')

# plot the bribe amount versus game result 
plot(R1,Amount,pch=19,col='blue')

# ...a better version of the plot
plot(jitter(R1),Amount,pch=19,col='blue')

# plot the game result versus player 1 points
plot(P1,jitter(R1),pch=19,col='blue')

T<-factor(R1==1) # Tie indicator
# plot tie indicator vs. player 1 points
plot(P1,jitter(as.numeric(T)),pch=19,col='blue')

# plot tie indicator vs. P1-P2
plot(P1-P2,jitter(as.numeric(T),amount=.1),pch=19,col='blue',
ylab='Jittered tie indicator (FALSE=1)')

# GLMs
#-------------------------------
g<-glm(T~P1,family=binomial)
g<-glm(T~P1+P2,family=binomial)
g<-glm(T~P1+P2+I(P1^2)+I(P2^2)+I(P1*P2),family=binomial)
g<-glm(T~I(P1^2)+I(P2^2)+I(P1*P2),family=binomial)
g<-glm(T~I(P1-P2)+I((P1-P2)^2),family=binomial)
g<-glm(T~I(P1-P2)+I((P1-P2)^2)+I((P1-P2)^3),family=binomial)
g<-glm(T~I((P1-P2)^2),family=binomial)


plot(P1-P2,predict(g,data.frame(P1=P1,P2=P2),type='response'),
pch=19,col='blue',ylab='Probability of a tie')
grid()

