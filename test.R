library(glmnet)
source("./utils.R")
source("./feature.R")

meta_ch = read.csv("./test/CmetricsTest.csv")
meta_jp = read.csv("./test/JmetricsTest.csv")
feature_name = c("ttr_mean","ttr_sd","ttr_norm","ttr_cumsum",
                 "ent_mean","ent_sd","ent_norm","ent_cumsum",
                 "con_ent_mean","con_ent_sd","con_ent_cumsum",
                 "dist_kl_mean","dist_kl_sd","dist_kl_cumsum",
                 "dist_cos_mean","dist_cos_sd","dist_cos_cumsum",
                 "cross_half_l2","cross_half_tfidf",
                 "cross_half_l2_nostopword","cross_half_tfidf_nostopword",
                 "thought","pronoun_first","pronoun_third")
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
chunksize = 500

pb = txtProgressBar(style=3)
for (i in 1:nrow(meta_ch)) {
	text = scan(paste0("~/github/NovelProject/data/chinese/",meta_ch$file_id_name[i]), what="character", sep=" ", quote="", quiet=TRUE)
	text = text[text!=""]
	meta_ch[i,c("ttr_mean","ttr_sd","ttr_norm","ttr_cumsum")] = feature_ttr(text, chunksize=chunksize, normalize=TRUE, cumsum=TRUE, replicate=500)
	meta_ch[i,c("ent_mean","ent_sd","ent_norm","ent_cumsum")] = feature_entropy(text, chunksize=chunksize, normalize=TRUE, cumsum=TRUE, replicate=500)
	meta_ch[i,c("con_ent_mean","con_ent_sd","con_ent_cumsum")] = feature_con_ent(text, chunksize=chunksize, cumsum=TRUE, replicate=500)
	meta_ch[i,c("dist_kl_mean","dist_kl_sd","dist_cos_mean","dist_cos_sd","dist_kl_cumsum","dist_cos_cumsum")] = feature_dist_to_global(text,nchunks=20)
	meta_ch[i,c("cross_half_l2","cross_half_tfidf")] = feature_cross_half_score(text,nchunks=20)
	meta_ch[i,c("cross_half_l2_nostopword","cross_half_tfidf_nostopword")] = feature_cross_half_score(text,nchunks=20,remove=c(stopword$ch,punct$ch))
	meta_ch[i,"thought"] = mean(text %in% thought$ch)
	meta_ch[i,"pronoun_first"] = mean(text=="我")
	meta_ch[i,"pronoun_third"] = mean(text=="他"|text=="她")
	setTxtProgressBar(pb,i/nrow(meta_ch))
}
close(pb)

pb = txtProgressBar(style=3)
for (i in 1:nrow(meta_jp)) {
	text = scan(paste0("~/github/NovelProject/data/japanese/",meta_jp$file_id[i],".txt"), what="character", sep=" ", quote="", quiet=TRUE)
	text = text[text!=""]
	meta_jp[i,c("ttr_mean","ttr_sd","ttr_norm","ttr_cumsum")] = feature_ttr(text, chunksize=chunksize, normalize=TRUE, cumsum=TRUE, replicate=500)
	meta_jp[i,c("ent_mean","ent_sd","ent_norm","ent_cumsum")] = feature_entropy(text, chunksize=chunksize, normalize=TRUE, cumsum=TRUE, replicate=500)
	meta_jp[i,c("con_ent_mean","con_ent_sd","con_ent_cumsum")] = feature_con_ent(text, chunksize=chunksize, cumsum=TRUE, replicate=500)
	meta_jp[i,c("dist_kl_mean","dist_kl_sd","dist_cos_mean","dist_cos_sd","dist_kl_cumsum","dist_cos_cumsum")] = feature_dist_to_global(text,nchunks=20)
	meta_jp[i,c("cross_half_l2","cross_half_tfidf")] = feature_cross_half_score(text,nchunks=20)
	meta_jp[i,c("cross_half_l2_nostopword","cross_half_tfidf_nostopword")] = feature_cross_half_score(text,nchunks=20,remove=c(stopword$jp,punct$jp))
	meta_jp[i,"thought"] = mean(text %in% thought$jp)
	setTxtProgressBar(pb,i/nrow(meta_jp))
}
close(pb)





write.csv(meta_ch,file="features_ch.csv",row.names=FALSE)
write.csv(meta_jp,file="features_jp.csv",row.names=FALSE)


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

confusion2 = matrix(0,3,3)
B = 100
for (i in 1:B) {
	romantic = sample(which(meta_ch$genre=="Romantic"))
	sr = sample(which(meta_ch$genre=="SR"))
	pop = sample(which(meta_ch$genre=="Pop"))
	
	ptrain = 4/5
	train = meta_ch[c(romantic[1:floor(length(romantic)*3/4)],sr[1:floor(length(sr)*3/4)],pop[1:floor(length(pop)*3/4)]),]
	#train = meta_ch[c(romantic[1:floor(length(romantic)*3/4)],pop[1:floor(length(pop)*3/4)]),]
	test = meta_ch[c(romantic[(floor(length(romantic)*3/4)+1):length(romantic)],sr[(floor(length(sr)*3/4)+1):length(sr)],pop[(floor(length(pop)*3/4)+1):length(pop)]),]
	#test = meta_ch[c(romantic[(floor(length(romantic)*3/4)+1):length(romantic)],pop[(floor(length(pop)*3/4)+1):length(pop)]),]
	features = c(6,7,9,11)

	fit = cv.glmnet(as.matrix(train[,features]),as.matrix(train[,3]),family="multinomial",type.multinomial = "grouped")
	true = test$genre
	pred = predict(fit,newx=as.matrix(test[,features]),s="lambda.min",type="class")
	confusion2 = confusion2 + table(as.character(true),pred)
	cat(i)
}
confusion2/B

fit = cv.glmnet(as.matrix(meta_ch[,features]),as.matrix(meta_ch[,3]),family="multinomial",type.multinomial="grouped", parallel=TRUE)