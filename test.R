library(glmnet)
source("./utils.R")
source("./feature.R")

meta_ch = read.csv("./test/CmetricsTest.csv")
meta_jp = read.csv("./test/JmetricsTest.csv")
feature_name = c("ttr_mean","ttr_sd","ttr_norm","ttr_cumsum",
                 "ent_mean","ent_sd","ent_norm","ent_cumsum",
                 "con_ent_mean","con_ent_sd","con_ent_cumsum",
                 "con_ent2_mean","con_ent2_sd",
                 "dist_kl_mean","dist_kl_sd","dist_kl_cumsum",
                 "dist_cos_mean","dist_cos_sd","dist_cos_cumsum",
                 "cross_half_l2","cross_half_tfidf",
                 "cross_half_l2_nostopword","cross_half_tfidf_nostopword",
                 "thought","pronoun_first","pronoun_third",
                 "period","punct","stopword")
meta_ch = cbind(meta_ch,matrix(0,nrow=nrow(meta_ch),ncol=length(feature_name),dimnames=list(NULL,feature_name)))
meta_jp = cbind(meta_jp,matrix(0,nrow=nrow(meta_jp),ncol=length(feature_name),dimnames=list(NULL,feature_name)))
stopword = list()
stopword$ch = scan("./misc/stopword_ch.txt", what="character", sep=" ")
stopword$jp = scan("./misc/stopword_jp.txt", what="character", sep=" ")
thought = list()
thought$ch = c("想","觉得","知道","心里","晓得","精神","想起","感到","觉","感觉","思想","感情")
thought$jp = c("思","考","気持")
punct = list()
punct$ch = scan("./misc/punct_ch.txt", what="character", sep=" ")
punct$jp = scan("./misc/punct_jp.txt", what="character", sep=" ")
chunksize = 1000

cat("Generating features for the Chinese corpus...\n")
pb = txtProgressBar(style=3)
record_ch = data.frame(ttr=NULL,ent=NULL,con_ent=NULL)
for (i in 1:nrow(meta_ch)) {
	text = scan(paste0("~/github/NovelProject/data/chinese/",meta_ch$file_id_name[i]), what="character", sep=" ", quote="", quiet=TRUE)
	text = text[text!=""]
	meta_ch[i,c("ttr_mean","ttr_sd","ttr_norm","ttr_cumsum")] = feature_ttr(text, chunksize=chunksize, remove=punct$ch, normalize=TRUE, cumsum=TRUE, replicate=500)
	meta_ch[i,c("ent_mean","ent_sd","ent_norm","ent_cumsum")] = feature_entropy(text, chunksize=chunksize, remove=punct$ch, normalize=TRUE, cumsum=TRUE, replicate=500)
	meta_ch[i,c("con_ent_mean","con_ent_sd","con_ent_cumsum")] = feature_con_ent(text, chunksize=chunksize, remove=punct$ch, cumsum=TRUE, replicate=500)
	meta_ch[i,c("con_ent2_mean","con_ent2_sd")] = feature_con_ent_new(text, chunksize=chunksize, remove=punct$ch)
	meta_ch[i,c("dist_kl_mean","dist_kl_sd","dist_cos_mean","dist_cos_sd","dist_kl_cumsum","dist_cos_cumsum")] = feature_dist_to_global(text, nchunks=20, remove=punct$ch)
	meta_ch[i,c("cross_half_l2","cross_half_tfidf")] = feature_cross_half_score(text, nchunks=20, remove=punct$ch)
	meta_ch[i,c("cross_half_l2_nostopword","cross_half_tfidf_nostopword")] = feature_cross_half_score(text,nchunks=20,remove=c(stopword$ch,punct$ch))
	meta_ch[i,"thought"] = mean(text %in% thought$ch)
	meta_ch[i,"pronoun_first"] = mean(text=="我")
	meta_ch[i,"pronoun_third"] = mean(text=="他"|text=="她")
	meta_ch[i,"period"] = mean(text=="。")
	meta_ch[i,"punct"] = mean(text %in% punct$ch)
	meta_ch[i,"stopword"] = mean(text %in% stopword$ch)
	setTxtProgressBar(pb,i/nrow(meta_ch))
}
close(pb)

cat("Recording chunkwise measurements...\n")
pb = txtProgressBar(style=3)
record_ch = data.frame(file_id=NULL,chunk_no=NULL,ttr=NULL,ent=NULL,con_ent=NULL)
for (i in 1:nrow(meta_ch)) {
	text = scan(paste0("~/github/NovelProject/data/chinese/",meta_ch$file_id_name[i]), what="character", sep=" ", quote="", quiet=TRUE)
	text = text[text!=""]
	ttr = feature_ttr(text, chunksize=chunksize, remove=punct$ch, return.value=TRUE)
	ent = feature_entropy(text, chunksize=chunksize, remove=punct$ch, return.value=TRUE)
	con_ent = feature_con_ent_new(text, chunksize=chunksize, remove=punct$ch, return.value=TRUE)
	record_ch = rbind(record_ch,data.frame(file_id=rep(meta_ch$file_id_name[i],length(ttr)),chunk_no=1:length(ttr),ttr=ttr,ent=ent,con_ent=con_ent))
	setTxtProgressBar(pb,i/nrow(meta_ch))
}
close(pb)

cat("Generating features for the Japanese corpus...\n")
pb = txtProgressBar(style=3)
for (i in 1:nrow(meta_jp)) {
	text = scan(paste0("~/github/NovelProject/data/japanese/",meta_jp$file_id[i],".txt"), what="character", sep=" ", quote="", quiet=TRUE)
	text = text[text!=""]
	meta_jp[i,c("ttr_mean","ttr_sd","ttr_norm","ttr_cumsum")] = feature_ttr(text, chunksize=chunksize, remove=punct$jp, normalize=TRUE, cumsum=TRUE, replicate=500)
	meta_jp[i,c("ent_mean","ent_sd","ent_norm","ent_cumsum")] = feature_entropy(text, chunksize=chunksize, remove=punct$jp, normalize=TRUE, cumsum=TRUE, replicate=500)
	meta_jp[i,c("con_ent_mean","con_ent_sd","con_ent_cumsum")] = feature_con_ent(text, chunksize=chunksize, remove=punct$jp, cumsum=TRUE, replicate=500)
	meta_jp[i,c("con_ent2_mean","con_ent2_sd")] = feature_con_ent_new(text, chunksize=chunksize, remove=punct$jp)
	meta_jp[i,c("dist_kl_mean","dist_kl_sd","dist_cos_mean","dist_cos_sd","dist_kl_cumsum","dist_cos_cumsum")] = feature_dist_to_global(text, nchunks=20, remove=punct$jp)
	meta_jp[i,c("cross_half_l2","cross_half_tfidf")] = feature_cross_half_score(text, nchunks=20, remove=punct$jp)
	meta_jp[i,c("cross_half_l2_nostopword","cross_half_tfidf_nostopword")] = feature_cross_half_score(text,nchunks=20,remove=c(stopword$jp,punct$jp))
	meta_jp[i,"thought"] = mean(text %in% thought$jp)
	#meta_jp[i,"pronoun_first"] = mean(text=="我")
	#meta_jp[i,"pronoun_third"] = mean(text=="他"|text=="她")
	meta_jp[i,"period"] = mean(text=="。")
	meta_jp[i,"punct"] = mean(text %in% punct$jp)
	meta_jp[i,"stopword"] = mean(text %in% stopword$jp)
	setTxtProgressBar(pb,i/nrow(meta_jp))
}
close(pb)

cat("Recording chunkwise measurements...\n")
pb = txtProgressBar(style=3)
record_jp = data.frame(file_id=NULL,chunk_no=NULL,ttr=NULL,ent=NULL,con_ent=NULL)
for (i in 1:nrow(meta_jp)) {
	text = scan(paste0("~/github/NovelProject/data/chinese/",meta_jp$file_id_name[i]), what="character", sep=" ", quote="", quiet=TRUE)
	text = text[text!=""]
	ttr = feature_ttr(text, chunksize=chunksize, remove=punct$jp, return.value=TRUE)
	ent = feature_entropy(text, chunksize=chunksize, remove=punct$jp, return.value=TRUE)
	con_ent = feature_con_ent_new(text, chunksize=chunksize, remove=punct$jp, return.value=TRUE)
	record_jp = rbind(record_jp,data.frame(file_id=rep(meta_jp$file_id_name[i],length(ttr)),chunk_no=1:length(ttr),ttr=ttr,ent=ent,con_ent=con_ent))
	setTxtProgressBar(pb,i/nrow(meta_jp))
}
close(pb)

write.csv(meta_ch,file="features_ch.csv",row.names=FALSE)
write.csv(record_ch,file="record_ch.csv",row.names=FALSE)
write.csv(meta_jp,file="features_jp.csv",row.names=FALSE)
write.csv(record_jp,file="record_jp.csv",row.names=FALSE)


confusion = matrix(0,2,2)
B = 100
for (i in 1:B) {
	control = sample(which(meta_jp$genre=="CONTROL"))
	prolet = sample(which(meta_jp$genre=="PROLET"))
	shishosetsu = sample(which(meta_jp$genre=="SHISHOSETSU"))
	
	ptrain = 4/5
	#train = meta_jp[c(control[1:floor(length(control)*3/4)],prolet[1:floor(length(prolet)*3/4)],shishosetsu[1:floor(length(shishosetsu)*3/4)]),]
	train = meta_jp[c(control[1:floor(length(control)*3/4)],shishosetsu[1:floor(length(shishosetsu)*3/4)]),]
	#test = meta_jp[c(control[(floor(length(control)*3/4)+1):length(control)],prolet[(floor(length(prolet)*3/4)+1):length(prolet)],shishosetsu[(floor(length(shishosetsu)*3/4)+1):length(shishosetsu)]),]
	test = meta_jp[c(control[(floor(length(control)*3/4)+1):length(control)],shishosetsu[(floor(length(shishosetsu)*3/4)+1):length(shishosetsu)]),]
	features = 9:26

	fit = cv.glmnet(as.matrix(train[,features]),as.matrix(train[,7]),family="binomial")
	true = test$genre
	pred = predict(fit,newx=as.matrix(test[,features]),s="lambda.min",type="class")
	confusion = confusion + table(as.character(true),pred)
	cat(i)
}
confusion/B

confusion_ch = matrix(0,3,3)
B = 100
for (i in 1:B) {
	romantic = sample(which(meta_ch$genre=="Romantic"))
	sr = sample(which(meta_ch$genre=="SR"))
	pop = sample(which(meta_ch$genre=="Pop"))
	
	ptrain = 9/10
	train = meta_ch[c(romantic[1:floor(length(romantic)*ptrain)],sr[1:floor(length(sr)*ptrain)],pop[1:floor(length(pop)*ptrain)]),]
	test = meta_ch[c(romantic[(floor(length(romantic)*ptrain)+1):length(romantic)],sr[(floor(length(sr)*ptrain)+1):length(sr)],pop[(floor(length(pop)*ptrain)+1):length(pop)]),]
	features = c("kl_score","ttr_mean","ent_mean","con_ent_mean","thought","pronoun_first","pronoun_third")

	fit = cv.glmnet(as.matrix(train[,features]),as.matrix(train[,"genre"]),family="multinomial",type.multinomial = "grouped",parallel=TRUE)
	true = test$genre
	pred = predict(fit,newx=as.matrix(test[,features]),s="lambda.min",type="class")
	confusion_ch = confusion_ch + table(as.character(true),pred)
	cat(i)
}
confusion_ch/B
#            Pop Romantic   SR
#  Pop      6.16     0.03 0.81
#  Romantic 0.57     8.72 3.71
#  SR       0.35     3.73 7.92

fit = cv.glmnet(as.matrix(meta_ch[,features]),as.matrix(meta_ch[,3]),family="multinomial",type.multinomial="grouped", parallel=TRUE)