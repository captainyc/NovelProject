chunk <- function(x, nchunks=NULL, chunksize=NULL, remove=NULL, randomize=FALSE, blocksize=1) {
	n = length(x)
	if (!is.null(nchunks)) chunksize = floor(n/nchunks)
	else if (!is.null(chunksize)) nchunks = floor(n/chunksize)
	else nchunks = 20; chunksize = floor(n/nchunks)
	if (randomize) {
		nblocks = ceiling(length(x)/blocksize)
		x = unlist(lapply(sample(nblocks),function(i) x[((i-1)*blocksize+1):min(i*blocksize,n)]))
	}
	chunks = lapply(1:nchunks,function(i) x[((i-1)*chunksize+1):(i*chunksize)])
	if (!is.null(remove)) {
		for (i in 1:length(chunks)) {
			chunks[[i]] = chunks[[i]][!(chunks[[i]] %in% remove)]
		}
	}
	return(chunks)
}

kl_div <- function(u, v){
	return(sum(u*log(u/v)))
}

cos_dist <- function(u, v) {
	return(1-sum(u*v)/(sqrt(sum(u^2))*sqrt(sum(v^2))))
}

dtm <- function(x) {
	vocab = sort(unique(unlist(x)))
	w = matrix(0,nrow=length(x),ncol=length(vocab),dimnames=list(NULL,vocab))
	for (i in 1:length(x)) {
		freq = table(x[[i]])/length(x[[i]])
		w[i,names(freq)] = freq
	}
	return(w)
}

tfidf <- function(x) {
	return(t(t(x)*log(nrow(x)/colSums(x>0))))
}

cross_half_score <- function(x,method="euclidean") {
	if (method=="euclidean") {
		w = as.matrix(dist(x, method="euclidean"))
	}
	else if (method=="cosine") {
		w = 1-x%*%t(x)/(sqrt(rowSums(x^2) %*% t(rowSums(x^2))))
	}
	else {
		cat('Invalid method.')
		return()
	}
	n = nrow(x)
	return(sum(w[1:floor(n/2),(floor(n/2)+1):n])/floor(n/2)^2)
}