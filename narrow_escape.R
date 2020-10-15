data = read.csv("C:\\Users\\alex\\Documents\\myR\\random walks\\sample.csv",sep = ",")
data = as.numeric(data[,1])
data
shapes <- seq(1.35,1.4, by = 0.001) 
scales <- seq(175,200,by = 0.5)
c = 0
i = 0
for (scale in scales){
	for (shape in shapes){
		i = i + 1
		gamma <- rgamma(length(data),shape = shape,scale = scale)
		c[i] <- ks.test(data,gamma)$p.value	
		if (i == 916){
			d <- shape
			e <- scale
		}
	}
	if ( i > 920){
		break
	}
}

c
which.max(c)
d
e

gamma <- rgamma(1000,shape = 1.3978,scale = 187.1)
ks.test(data,gamma)
plot(ecdf(x = data), main = "ECDF of x and y", lwd=1)
lines(ecdf(x = gamma), col = 2)

library(fitdistrplus)
plotdist(data, histo = TRUE, demp = TRUE,pch = 19)
descdist(data, boot = 500)


fg <- fitdist(data, "gamma")
fln <- fitdist(data, "lnorm")
fw <- fitdist(data, "weibull")


plot.legend <- c("Weibull","lognormal", "gamma")

denscomp(list(fw,fln, fg), legendtext = plot.legend)
qqcomp(list(fw,fln, fg), legendtext = plot.legend)
cdfcomp(list(fw,fln, fg), legendtext = plot.legend)
ppcomp(list(fw,fln, fg), legendtext = plot.legend)

gammaboot <- bootdist(fg, niter = 501)
summary(gammaboot)
plot(gammaboot)




