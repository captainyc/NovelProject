source('./utils.R')

# Japanese
# put the tokenized japanese text files in the directory <path>
path = './data/japanese/'
record_jp = read.csv('./record_jp.csv')
head(record_jp)
# chunk with the 10 highest conditional entropy
ind = sort(record_jp$con_ent,decreasing=TRUE,index.return=TRUE)$ix[1:10]
for (i in ind) {
	temp = record_jp[i,]
	get_chunk(paste0(path,temp$file_id,'.txt'),temp$chunk_no,chunksize=1000)
	cat('\n\n')
}
# chunk with the 10 lowest conditional entropy
ind = sort(record_jp$con_ent,index.return=TRUE)$ix[1:10]
for (i in ind) {
	temp = record_jp[i,]
	get_chunk(paste0(path,temp$file_id,'.txt'),temp$chunk_no,chunksize=1000)
	cat('\n\n')
}

# Chinese
path = './data/chinese/'
record_ch = read.csv('./record_ch.csv')
head(record_ch)
# chunk with the 10 highest conditional entropy
ind = sort(record_ch$con_ent,decreasing=TRUE,index.return=TRUE)$ix[1:10]
for (i in ind) {
	temp = record_ch[i,]
	get_chunk(paste0(path,temp$file_id),temp$chunk_no,chunksize=1000)
	cat('\n\n')
}
# chunk with the 10 lowest conditional entropy
ind = sort(record_ch$con_ent,index.return=TRUE)$ix[1:10]
for (i in ind) {
	temp = record_ch[i,]
	get_chunk(paste0(path,temp$file_id),temp$chunk_no,chunksize=1000)
	cat('\n\n')
}
