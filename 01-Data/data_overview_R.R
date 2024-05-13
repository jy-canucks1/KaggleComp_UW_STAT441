############################# LOAD DATA #####################
X_train <- read.csv("01-Data/X_train.csv", header=T)
y_train <- read.csv("01-Data/y_train.csv", header=T)
X_test <- read.csv("01-Data/X_test.csv", header=T)


########################### TABLIZE ##########################
options(max.print = 99999999)
tablize <- function(data, filename) {
  cname <- colnames(data) 
  for (idx in 2:ncol(data)){
    cat(cname[idx], file=filename, append=T)
    cat("\n", file=filename, append=T)
    tmp <- data.frame(table=(table(data[,idx])))
    write.table(tmp, file=filename, append=T, sep='\t', 
                row.names=F, col.names=F)
    cat("\n", file=filename, append=T)
  }
}
tablize(X_train, "Train.txt")
tablize(X_test, "Test.txt")

########################################################################

var <- c('v133_11c', 'v134_11c', 'v135_11c', 'v136_11c', 'v137_11c', 'v138_11c', 
         'v139_11c', 'v140_11c', 'v141_11c')
var_org <- c('v133', 'v134', 'v135', 'v136', 'v137', 'v138', 'v139', 'v140', 'v141')

######################## ACCESS ##############################
access <- function(var1, var2) {
  for (i in 1:9) {
    print(paste(var_org[i],"vs", var[i]))
    print(table(X_train[[var1[i]]], X_train[[var2[i]]]))
  }
}
access(var, var_org)

######################### SWAP ##############################
swap <- function(var1, var2) {
  for (i in 1:9) {
    d1 <- X_train[[var1[i]]]
    d2 <- X_train[[var2[i]]]
    d1[d1 == -4] <- d2[d1 == -4]
  }
}
swap(var, var_org)
############################################################





