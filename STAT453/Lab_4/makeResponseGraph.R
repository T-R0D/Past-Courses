windows()
plot(c(23,30),c(0,1),type='n',
     xlab='Width (cm)',ylab='Proportion Having Satellite',log='x')
text(width,propSat[1:8],labels=as.character(ageCat[1:8]),col='blue')
text(width,propSat[9:16],labels=as.character(ageCat[9:16]),col='red')

# Proportion: Prediction plot
#======================================
logitWidth<-seq(22.5,30.5,0.1)
lines(logitWidth,predict(model5,
                   data.frame(width=logitWidth,ageCat=factor(rep("Y",length(logitWidth)),levels=levels(ageCat))),
                   type='response'),col='blue')
lines(logitWidth,predict(model5,
                   data.frame(width=logitWidth,ageCat=factor(rep("O",length(logitWidth)),levels=levels(ageCat))),
                   type='response'),col='red')
