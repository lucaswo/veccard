require(ggplot2)
require(Cairo)

errors <- read.table("all_qerror.csv", header=FALSE)
outputfilename <- "boxplot.pdf"

plot <- ggplot(errors, aes(x="", y=V1)) +
  geom_boxplot() +
  scale_y_continuous(trans="log10") +
  ylab("Q-Error") +
  xlab("Estimator")

CairoPDF(outputfilename)
print(plot)
dev.off()

cat("\n\nSUMMARY\n")
print(summary(errors$V1))
cat("\n\nQUANTILES\n")
quants <- quantile(errors$V1, probs=c(0, .01, 0.05, .1, .25, .5, .75, .90, .95, .99, 1))
print(quants)


cat("\nOutput written to", outputfilename, "\n")
