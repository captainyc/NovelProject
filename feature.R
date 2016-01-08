feature_ttr <- function(text, nchunks=NULL, chunksize=NULL, remove=NULL, normalize=FALSE, cumsum=FALSE, replicate=100, return.value=FALSE) {
	chunks = chunk(text, nchunks, chunksize, remove)
	TTR = sapply(1:length(chunks),function(i) length(unique(chunks[[i]]))/length(chunks[[i]]))
	if (return.value) return(TTR)
	features = data.frame(mean=mean(TTR),sd=sd(TTR))
	if (normalize) {
		TTR_norm = mean(sapply(1:replicate,function(i) length(unique(sample(text,length(chunks[[1]]))))/length(chunks[[1]])))
		features$norm = mean(TTR)-TTR_norm
	}
	if (cumsum) features$cumsum = cumsum_test(TTR,replicate=replicate)
	return(features)
}

feature_entropy <- function(text, nchunks=NULL, chunksize=NULL, remove=NULL, normalize=FALSE, cumsum=FALSE, replicate=100, return.value=FALSE) {
	chunks = chunk(text, nchunks, chunksize, remove)
	entropy = sapply(1:length(chunks),function(i) {temp = table(chunks[[i]])/length(chunks[[i]]);sum(-temp*log(temp))} )
	if (return.value) return(entropy)
	features = data.frame(mean=mean(entropy),sd=sd(entropy))
	if (normalize) {
		entropy_norm = mean(sapply(1:replicate, function(i) { temp = table(sample(text,length(chunks[[1]])))/length(chunks[[1]]); sum(-temp*log(temp)) }))
		features$norm = (mean(entropy)-entropy_norm)
	}
	if (cumsum) features$cumsum = cumsum_test(entropy,replicate=replicate)
	return(features)
}

feature_con_ent <- function(text, nchunks=NULL, chunksize=NULL, remove=NULL, cumsum=FALSE, replicate=100, return.value=FALSE) {
	chunks = chunk(text, nchunks, chunksize, remove)
	conditional_entropy = numeric(length(chunks))
	eps = 1e-10
	for (i in 1:length(chunks)) {
		pair_count = table(chunks[[i]][-length(chunks[[i]])],chunks[[i]][-1])
		conditional_entropy[i] = -sum(log(pair_count/rowSums(pair_count)+eps)*(pair_count/sum(pair_count)))
	}
	if (return.value) return(conditional_entropy)
	features = data.frame(mean=mean(conditional_entropy),sd=sd(conditional_entropy))
	if (cumsum) features$cumsum = cumsum_test(conditional_entropy,replicate=replicate)
	return(features)
}

feature_con_ent_new <- function(text, nchunks=NULL, chunksize=NULL, remove=NULL, cumsum=FALSE, replicate=100, return.value=FALSE) {
	text[text %in% remove] = "_"
	chunks = chunk(text, nchunks, chunksize)
	conditional_entropy = numeric(length(chunks))
	eps = 1e-10
	for (i in 1:length(chunks)) {
		pair_count = table(chunks[[i]][-length(chunks[[i]])],chunks[[i]][-1])
		pair_count[rownames(pair_count)!='_',colnames(pair_count)!='_']
		pair_count = pair_count[rowSums(pair_count)>0,]
		conditional_entropy[i] = -sum(log(pair_count/rowSums(pair_count)+eps)*(pair_count/sum(pair_count)))
	}
	if (return.value) return(conditional_entropy)
	features = data.frame(mean=mean(conditional_entropy),sd=sd(conditional_entropy))
	if (cumsum) features$cumsum = cumsum_test(conditional_entropy,replicate=replicate)
	return(features)
}

feature_dist_to_global <- function(text, nchunks=NULL, chunksize=NULL, remove=NULL, cumsum=FALSE, replicate=100, return.value=FALSE) {
	chunks = chunk(text, nchunks, chunksize, remove)
	global_vec = table(text)/length(text)
	KL = sapply(1:length(chunks),function(i) { freq=table(chunks[[i]])/length(chunks[[i]]); return(kl_div(freq,global_vec[names(freq)])) } )
	COS = sapply(1:length(chunks),function(i) { freq=table(chunks[[i]])/length(chunks[[i]]); return(cos_dist(freq,global_vec[names(freq)]))} )
	if (return.value) return(data.frame(kl=KL,cos=COS))
	features = data.frame(kl_mean=mean(KL),kl_sd=sd(KL),cos_mean=mean(COS),cos_sd=sd(COS))
	if (cumsum) features$kl_cumsum = cumsum_test(KL,replicate=replicate); features$cos_cumsum = cumsum_test(COS,replicate=replicate)
	return(features)
}

feature_cross_half_score <- function(text, nchunks=NULL, chunksize=NULL, remove=NULL) {
	chunks = chunk(text, nchunks, chunksize, remove)
	DTM = dtm(chunks)
	TFIDF = tfidf(DTM)
	return(c(cross_half_score(DTM,"euclidean"),cross_half_score(TFIDF,"cosine")))
}