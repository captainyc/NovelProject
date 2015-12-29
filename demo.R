library(glmnet)
source("./utils.R")
source("./feature.R")

text = list()
text[[1]] = scan("./data/sample_chinese.txt", what="character", sep=" ")
text[[2]] = scan("./data/sample_japanese.txt", what="character", sep=" ")
stopword = list()
stopword[[1]] = scan("./misc/stopword_ch.txt", what="character", sep=" ")
stopword[[2]] = scan("./misc/stopword_jp.txt", what="character", sep=" ")
punct = list()
punct[[1]] = scan("./misc/punct_ch.txt", what="character", sep=" ")
punct[[2]] = scan("./misc/punct_jp.txt", what="character", sep=" ")

chunksize = 1000
feature_name = c("ttr_mean","ttr_sd",
                 "ent_mean","ent_sd",
                 "con_ent_mean","con_ent_sd",
                 "dist_kl_mean","dist_kl_sd",
                 "dist_cos_mean","dist_cos_sd",
                 "cross_half_l2","cross_half_tfidf",
                 "cross_half_l2_nostopword","cross_half_tfidf_nostopword")
features = matrix(0, nrow=0, ncol=length(feature_name), dimnames=list(NULL,feature_name))

for (i in 1:length(text)) {
	features = rbind(features,numeric(length(feature_name)))
	features[i,c("ttr_mean","ttr_sd")] = feature_ttr(text[[i]], chunksize=chunksize)
	features[i,c("ent_mean","ent_sd")] = feature_entropy(text[[i]], chunksize=chunksize)
	features[i,c("con_ent_mean","con_ent_sd")] = feature_con_ent(text[[i]], chunksize=chunksize)
	features[i,c("dist_kl_mean","dist_kl_sd","dist_cos_mean","dist_cos_sd")] = feature_dist_to_global(text[[i]], chunksize=chunksize)
	features[i,c("cross_half_l2","cross_half_tfidf")] = feature_cross_half_score(text[[i]],nchunks=20)
	features[i,c("cross_half_l2_nostopword","cross_half_tfidf_nostopword")] = feature_cross_half_score(text[[i]],nchunks=20,remove=c(stopword[[i]],punct[[i]]))
}