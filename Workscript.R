
library("caret")
library("dplyr")
library(doMC)
registerDoMC(cores = 4)


if (!file.exists("./pml-training.csv")){
 
    fileurl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    download.file(fileurl,"./pml-training.csv",method="curl")
}

if (!file.exists("./pml-testing.csv")){
  
  fileurl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
  download.file(fileurl,"./pml-testing.csv",method="curl")
}

trainset <- read.csv(file="./pml-training.csv",header = TRUE);
testset  <- read.csv(file="./pml-testing.csv",header = TRUE);

removeredunt <- trainset %>% select(-X,-raw_timestamp_part_1,-raw_timestamp_part_2,-cvtd_timestamp,-new_window,-num_window)
#remove varible with all NA
withoutallna <- removeredunt[, colSums(is.na(removeredunt)) == 0]
dim(withoutallna)

nearzero <- nearZeroVar(withoutallna,saveMetrics = TRUE)
cleaned <- withoutallna[,nearzero[,"nzv"]==FALSE]
dim(cleaned)

#default seed
set.seed(1234)

inTraining <- createDataPartition(cleaned$classe, p = .7, list=FALSE)

trainsetwithoutna <- cleaned[inTraining,]
validation <- cleaned[-inTraining,]
dim(trainsetwithoutna)

pcadata <- preProcess(trainsetwithoutna %>% select(-classe),method="pca")
processtrainset <- predict(pcadata,trainsetwithoutna %>% select(-classe))

dim(processtrainset)

start <- Sys.time()
set.seed(1234)
model <- train(trainsetwithoutna$classe~.,data=processtrainset,method="rf",verbose=TRUE,allowparallel=TRUE,ntree=100,trControl=trainControl(method="cv",number="2"),prox=TRUE);
end <- Sys.time()

end - start 

confusionMatrix(model)

saveRDS(model,file="baserf.rds")



