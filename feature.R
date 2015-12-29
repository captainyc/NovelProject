feature_ttr <- function(text, nchunks=NULL, chunksize=NULL, remove=NULL, normalize=FALSE, blocksize=1, replicate=100) {
	chunks = chunk(text, nchunks, chunksize, remove)
	TTR = sapply(1:length(chunks),function(i) length(unique(chunks[[i]]))/length(chunks[[i]]))
	features = c(mean(TTR),sd(TTR))
	if (normalize) {
		TTR_perm = numeric(replicate)
		for (j in 1:replicate) {
			chunks_temp = chunk(text, nchunks, chunksize, randomize=TRUE, blocksize=blocksize)
			TTR_perm[j] = mean(sapply(1:length(chunks_temp),function(i) length(unique(chunks_temp[[i]]))/length(chunks_temp[[i]])))
		}
		features = c(features,(mean(TTR)-mean(TTR_perm))/sd(TTR_perm))
	}
	return(features)
}

feature_entropy <- function(text, nchunks=NULL, chunksize=NULL, remove=NULL, normalize=FALSE, blocksize=1, replicate=100) {
	chunks = chunk(text, nchunks, chunksize, remove)
	entropy = sapply(1:length(chunks),function(i) {temp = table(chunks[[i]])/length(chunks[[i]]);sum(-temp*log(temp))} )
	features = c(mean(entropy),sd(entropy))
	if (normalize) {
		entropy_perm = numeric(replicate)
		for (j in 1:replicate) {
			chunks_temp = chunk(text, nchunks, chunksize, randomize=TRUE, blocksize=blocksize)
			entropy_perm[j] = mean(sapply(1:length(chunks_temp),function(i) {temp = table(chunks_temp[[i]])/length(chunks_temp[[i]]);sum(-temp*log(temp))} ))
		}
		features = c(features,(mean(entropy)-mean(entropy_perm))/sd(entropy_perm))
	}
	return(features)
}

feature_con_ent <- function(text, nchunks=NULL, chunksize=NULL) {
	chunks = chunk(text, nchunks, chunksize, remove)
	conditional_entropy = numeric(length(chunks))
	eps = 1e-10
	for (i in 1:length(chunks)) {
		pair_count = table(chunks[[i]][-length(chunks[[i]])],chunks[[i]][-1])
		conditional_entropy[i] = -sum(log(pair_count/rowSums(pair_count)+eps)*(pair_count/sum(pair_count)))
	}
	return(c(mean(conditional_entropy),sd(conditional_entropy)))
}

feature_dist_to_global <- function(text, nchunks=NULL, chunksize=NULL, remove=NULL) {
	chunks = chunk(text, nchunks, chunksize, remove)
	global_vec = table(text)/length(text)
	KL = sapply(1:length(chunks),function(i) { freq=table(chunks[[i]])/length(chunks[[i]]); return(kl_div(freq,global_vec[names(freq)])) } )
	COS = sapply(1:length(chunks),function(i) { freq=table(chunks[[i]])/length(chunks[[i]]); return(cos_dist(freq,global_vec[names(freq)]))} )
	return(c(mean(KL),sd(KL),mean(COS),sd(COS)))
}

feature_cross_half_score <- function(text, nchunks=NULL, chunksize=NULL, remove=NULL) {
	chunks = chunk(text, nchunks, chunksize, remove)
	DTM = dtm(chunks)
	TFIDF = tfidf(DTM)
	return(c(cross_half_score(DTM,"euclidean"),cross_half_score(TFIDF,"cosine")))
}