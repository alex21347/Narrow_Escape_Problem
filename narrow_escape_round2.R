data = read.csv("C:\\Users\\alex\\Documents\\myR\\random walks\\narrow_escape_pdf.csv",sep = ",")
data1 = as.numeric(data[,1])
data2 = as.numeric(data[,2])
data3 = as.numeric(data[,3])
data4 = as.numeric(data[,4])
data5 = as.numeric(data[,5])
data6 = as.numeric(data[,6])
data7 = as.numeric(data[,7])
data8 = as.numeric(data[,8])
data9 = as.numeric(data[,9])
data10 = as.numeric(data[,10])

data15 = read.csv("C:\\Users\\alex\\Documents\\myR\\random walks\\narrow_escape_pdf15.csv",sep = ",")
data15 = as.numeric(data15[,1])

gamma <- rgamma(1000,shape = 1.3978,scale = 187.1)
ks.test(data,gamma)
plot(ecdf(x = data1), main = "ECDF of x and y", lwd=1)
lines(ecdf(x = gamma), col = 2)

library(fitdistrplus)
plotdist(data1, histo = TRUE, demp = TRUE,pch = 19)
plotdist(data2, histo = TRUE, demp = TRUE,pch = 19)
plotdist(data10, histo = TRUE, demp = TRUE,pch = 19)
plotdist(data8, histo = TRUE, demp = TRUE,pch = 19)
plotdist(data15, histo = TRUE, demp = TRUE,pch = 19)

descdist(data1, boot = 500)
descdist(data2, boot = 500)
descdist(data3, boot = 500)
descdist(data4, boot = 500)
descdist(data5, boot = 500)
descdist(data6, boot = 500)
descdist(data7, boot = 500)
descdist(data8, boot = 500)
descdist(data9, boot = 500)
descdist(data10, boot = 500)
descdist(data15, boot = 500)



fg <- fitdist(data3, "gamma")
fb <- fitdist(data3, "beta")
fln <- fitdist(data3, "lnorm")
fw <- fitdist(data3, "weibull")


plot.legend <- c("Weibull","lognormal", "gamma")

denscomp(list(fw,fln, fg), legendtext = plot.legend)
qqcomp(list(fw,fln, fg), legendtext = plot.legend)
cdfcomp(list(fw,fln, fg), legendtext = plot.legend)
ppcomp(list(fw,fln, fg), legendtext = plot.legend)

gammaboot <- bootdist(fg, niter = 501)
summary(gammaboot)
plot(gammaboot)
