#=========================================
#              STAT 453/653
#        Generalized Linear Models:
#      Several independent variables
#=========================================

# Set-up
#=========================================
#ldose<-rep(0:5,2)
#numdead<-c(1,4,9,13,18,20,0,2,6,10,12,16)
#sex<-factor(rep(c("M","F"),c(6,6)))
#SF<-cbind(numdead,numalive=20-numdead)

yearsSince<-c( 4, 5, 6, 7, 8, 9,10,11,12,13,14,16,17,24,28,31,50, 4, 5, 6, 7, 8, 9,10,11,12,13,14,16,17,24,28,31,50)
   success<-c( 2, 6, 5, 3, 2, 2, 1, 1, 0, 1, 0, 2, 0, 0, 0, 1, 1, 6,14, 6, 4, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1)
  numAreas<-c( 5,13,15,13,11, 3, 2, 1, 0, 2, 1, 4, 0, 0, 0, 1, 8,11,23,15,12,11,10, 1, 2, 2, 0, 0, 0, 1, 2, 1, 0, 1)
SF<-cbind(success, (numAreas-success))
propSuccess<-c(success/numAreas)
state<-factor(rep(c("NV","O"),c(17,17)))

# GLM
#=========================================
model1<-glm(SF~yearsSince,family=binomial)
model2<-glm(SF~state,family=binomial)
model3<-glm(SF~state+yearsSince,family=binomial)
model4<-glm(SF~state:yearsSince,family=binomial)
model5<-glm(SF~state*yearsSince,family=binomial)

# Proportion: Data plot
#=======================================
windows()
#par(bg='yellow')
plot(c(4,50),c(0,1),type='n',
     xlab='Years Since Last Gather',ylab='Proportion of Successfully Managed Areas',log='x')
text(yearsSince,propSuccess[1:17],labels=as.character(state[1:17]),col='blue')
text(yearsSince,propSuccess[18:34],labels=as.character(state[18:34]),col='red')

# Proportion: Prediction plot
#======================================
lgather<-seq(4,50,0.1)
lines(lgather,predict(model5,
                   data.frame(yearsSince=lgather,state=factor(rep("NV",length(lgather)),levels=levels(state))),
                   type='response'),col='slategray4')
lines(lgather,predict(model5,
                   data.frame(yearsSince=lgather,state=factor(rep("O",length(lgather)),levels=levels(state))),
                   type='response'),col='red4')

# Logit: Data plot
#======================================
windows()
plot(c(1,32),c(-5,5),type='n',
     xlab='Dose, mg',ylab='Logit function',log='x')
text(2^ldose,log(numdead[1:6]/20/(1-numdead[1:6]/20)),
     labels=as.character(sex[1:6]),col='blue')
text(2^ldose,log(numdead[7:12]/20/(1-numdead[7:12]/20)),
     labels=as.character(sex[7:12]),col='red')

# Logit: Prediction plot
#======================================
ld<-seq(0,5,0.1)
p<-predict(g,
           data.frame(ldose=ld,sex=factor(rep("M",length(ld)),levels=levels(sex))),
           type='response')
lines(2^ld,log(p/(1-p)),col='blue')
p<-predict(g,
           data.frame(ldose=ld,sex=factor(rep("F",length(ld)),levels=levels(sex))),
           type='response')
lines(2^ld,log(p/(1-p)),col='red')
