# Logit: Data plot
#======================================
windows()
plot(c(23,30),c(-2.75,0.25),type='n',
     xlab='Width (cm)',ylab='Logit function',log='x')
text(width,log(propSat[1:8]),
     labels=as.character(ageCat[1:8]),col='blue')
text(width,log(propSat[9:16]),
     labels=as.character(ageCat[9:16]),col='red')

# Logit: Prediction plot
#======================================
logitWidth<-seq(22.5,30.5,0.1)
p<-predict(model4,
           data.frame(width=logitWidth,ageCat=factor(rep("Y",length(logitWidth)),levels=levels(ageCat))),
           type='response')
lines(logitWidth,log(p/(1-p)),col='blue')
p<-predict(model4,
           data.frame(width=logitWidth,ageCat=factor(rep("O",length(logitWidth)),levels=levels(ageCat))),
           type='response')
lines(logitWidth,log(p/(1-p)),col='red')
