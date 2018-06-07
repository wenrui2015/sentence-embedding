# -*- coding: utf-8 -*-
"""
Created on Thu May 17 10:08:19 2018

@author: ruiwen
"""
import numpy as np
import sys
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD
import gensim

#sicktest A SIMPLE BUT TOUGH-...       myalgo             bag-of-word             tfidf             ave-glove
#               0.72              0.6746507525749791    0.5573789676211157   0.6036350458703256  0.6918761221067486  
#sicktrain A SIMPLE BUT TOUGH-...     myalgo               bag-of-word            tfidf            ave-glove
#               0.73              0.6825154662029216    0.5582407172702628   0.6044042015550684  0.6940753917593857
#(no sub main com)0.715

#class task:
#my_algo:0.741900054914882(fre>8 train_file:sentiment-train-reduced)
#a simple but tough to beat:0.7468423942888522
#ave-glove 0.7616694124107634

broken_word=['at','to','email']                       
#load the word frequency dict                    
wordfre_dic=dict()
count=1
with open("data/res/enwiki_vocab_min200.txt",encoding="utf8")as fi:
    for line in fi:
        line=line.split()
        if len(line)!=3:
            continue
        wordfre_dic[line[0]]=float(line[2])
    fi.close()
print("wordfre_dic.shape",wordfre_dic['the'])


def get_diction(input_file,output_file):
    diction=dict()
    f_out=open(output_file,'w')
    wordvec_dic=gensim.models.KeyedVectors.load_word2vec_format("data/res/glovel_model.txt") 
    print("load the glove vector over!!!")
    with open(input_file)as fi:
        for line in fi:
            line=line.strip().split("\t")
            if len(line)!=2:#if len(line)!=3:
                continue
            sen1,_=line#sen1,sen2,_=line
            for ele in sen1.split():#for ele in sen1.split()+sen2.split():
                if ele in diction:
                    diction[ele]+=1
                else:
                    diction[ele]=1
    diction=sorted(diction.items(),key=lambda x:x[1],reverse=True)
    for key ,value in diction:
        if key in wordvec_dic:
            f_out.write(key+"\t"+str(value)+"\t"+" ".join([str(vec) for vec in wordvec_dic[key]])+"\n")
#get_diction(input_file="data/train_data/sentiment-train",output_file="data/res/wordfre_sentiment-train")        
def get_bigvec(input_file):
    diction=dict()
    bigvec=[]
    with open(input_file)as fi:
        for line in fi:
            line=line.strip().split("\t")
            if len(line)!=3:
                continue
            if int(line[1])>8:
                vec=[float(vec) for vec in line[2].strip().split()]
                bigvec.append(vec)
                diction[line[0]]=vec
        fi.close()
    return diction,np.asarray(bigvec).T

def my_sen2vec(sen,diction,bigvec):
    sen_vec=[]
    for word in sen.split():
        if word in diction:
            sen_vec.append(diction[word])
    if len(sen_vec)<1:
        return "empty"
    sen_vec=np.asarray(sen_vec);sen_vec=np.reshape(sen_vec,[sen_vec.shape[0],-1])
    sen_vec_dis=np.sqrt((sen_vec*sen_vec).sum(axis=1));sen_vec_dis=np.reshape(sen_vec_dis,[sen_vec_dis.shape[0],1])
    big_vec_dis=np.sqrt((bigvec*bigvec).sum(axis=0))
    middle=np.dot(sen_vec,bigvec)/sen_vec_dis/big_vec_dis
    result=np.max(middle,axis=0)
    #print (result)
    return result
def my_sens2vec(sens,wordfre_file="data/res/wordfre"):
    label_list=[]
    diction,bigvec=get_bigvec(input_file=wordfre_file)
    print("bigvec.shape",(bigvec.shape))
    sens_vec=[]
    count=0
    for sen in sens:
        count+=1
        if count % 5000==0:
            print(count)
        tmp=sen.split("\t")
        if len(tmp)==2:#with label
            sen,label=tmp
            sen_vec=my_sen2vec(sen,diction,bigvec) 
            if sen_vec=="empty":
                continue
            else:
                sen_vec=my_sen2vec(sen,diction,bigvec) 
                sens_vec.append(sen_vec)
                label_list.append(label)    
        else:
            sen_vec=my_sen2vec(sen,diction,bigvec) 
            sens_vec.append(sen_vec)
    sens_vec=np.asarray(sens_vec)
    print("sens_vec.shape",sens_vec.shape)
    return sens_vec,label_list
def my_algo():
    sen1_vec,_=my_sens2vec(sen1_list);sen1_vec=reduce_singular(sen1_vec,npc=0)
    sen2_vec,_=my_sens2vec(sen2_list);sen2_vec=reduce_singular(sen2_vec,npc=0)
    print("sen1_vec.shape",sen1_vec.shape) 
    print("sen2_vec.shape",sen2_vec.shape) 
    similarity=cosin_simi(sen1_vec,sen2_vec)
    np.save("data/result/my_similarity.npy",similarity)
    p_coef,p_value=pearsonr(similarity.tolist(),label_list)
    print (p_coef,p_value)
    return sen1_vec,sen2_vec,similarity
def wr_sen2vec(sen,wordvec_dic,a=1e-3,is_ave=False):
    wordvec_list=[]
    word_list=[]
    wordfre_list=[]
    for word in sen.split():
        if  word in wordfre_dic and word in wordvec_dic:
            wordvec_list.append(wordvec_dic[word])
            word_list.append(word)
            wordfre_list.append(wordfre_dic[word])
        else:
            pass
    if len(wordfre_list)>0:
        wordfre_list=np.asarray(wordfre_list);wordfre_list=np.reshape(wordfre_list,[wordfre_list.shape[0],-1])
        weights=a/(a+wordfre_list)/len(word_list)
        wordvec_list=np.asarray(wordvec_list)
        if is_ave:
            vec=(np.sum(wordvec_list,axis=0)/len(word_list)).tolist()
        else:
            vec=np.sum(weights*wordvec_list,axis=0).tolist()
        return vec
    else:
        print (sen+"  's vector is empty for can't find the a word in dictionary")
        return "empty"
def reduce_singular(vec,npc):
    #calculate the first singular vector
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)#only remain the first singular value
    u=svd.fit(vec).components_
    #vs = vs -u x uT x vs
    if npc==1:
        vec = vec - vec.dot(u.transpose()) * u
    else:
        vec = vec - vec.dot(u.transpose()).dot(u)
    return vec
def wr_sens2vec(sens,wordvec_dic,npc=1,a=1e-3):
    sens_vec=[]
    label_list=[]
    for sen in sens:
        tmp=sen.strip().split("\t")
        if len(tmp)==2:
            sen,label=tmp
            vec=wr_sen2vec(sen,wordvec_dic,a)
            if vec=="empty":
                continue
            else:
                vec=wr_sen2vec(sen,wordvec_dic,a)
                sens_vec.append(vec)
                label_list.append(label)
        else:
            vec=wr_sen2vec(sen,wordvec_dic,a)
            sens_vec.append(vec)
    #sens_vec:[sens,embedding_size]
    sens_vec=np.asarray(sens_vec)
    sens_vec=reduce_singular(sens_vec,npc)
    print("sens_vec.shape",sens_vec.shape)
    return sens_vec,label_list
def cosin_simi(vec1,vec2):
    inner=(vec1*vec2).sum(axis=1)
    sen1_dis=np.sqrt((vec1*vec1).sum(axis=1))
    sen2_dis=np.sqrt((vec2*vec2).sum(axis=1))
    similarity=inner/sen1_dis/sen2_dis
    return similarity
def wr_algo(is_ver=False):
    wordvec_dic=gensim.models.KeyedVectors.load_word2vec_format("data/res/glovel_model.txt") 
    vocab=wordvec_dic.vocab 
    print("load the glove vector over!!!")  
    print("the vocab size is:",len(vocab))  
    sen1_vec,_=wr_sens2vec(sen1_list,wordvec_dic,npc=1,a=1e-3,is_ave=is_ver)  
    sen2_vec,_=wr_sens2vec(sen2_list,wordvec_dic,npc=1,a=1e-3,is_ave=is_ver) 
    print("sen1_vec.shape",sen1_vec.shape) 
    print("sen2_vec.shape",sen2_vec.shape) 
    similarity=cosin_simi(sen1_vec,sen2_vec)
    np.save("data/result/similarity.npy",similarity)
    p_coef,p_value=pearsonr(similarity.tolist(),label_list)
    print(p_coef,p_value)

def tf_idf_vec(sen_corpus,dictionary=None):
    if not dictionary:
        dictionary = corpora.Dictionary(sen_corpus)  # 构建语料库词典
    sen_corpus = [dictionary.doc2bow(text) for text in sen_corpus]
    tfidf = models.TfidfModel(sen_corpus)
    sen_tfidf = tfidf[sen_corpus]
    sen_tfidf = gensim.matutils.corpus2dense(sen_tfidf, num_terms=len(dictionary.keys())).T
    return sen_tfidf,dictionary
def tf_idf():
    sen1_corpus = [[token for token in sen.split() ] for sen in sen1_list]  
    sen2_corpus = [[token for token in sen.split()] for sen in sen2_list]  
    sen_corpus=sen1_corpus+sen2_corpus
    
    sen_tfidf,dictionary=tf_idf_vec(sen_corpus,dictionary=None)
    
    #dictionary = corpora.Dictionary(sen_corpus)  # 构建语料库词典
    cow=sen_tfidf.shape[0]
    index1=range(0,int(cow/2));index2=range(int(cow/2),cow)
    sen1_vec=sen_tfidf[index1];sen2_vec=sen_tfidf[index2]
    print(sen1_vec[0],sen1_vec[1])
    similarity=cosin_simi(sen1_vec,sen2_vec)
    p_coef,p_value=pearsonr(similarity.tolist(),label_list)
    print(p_coef,p_value)
    
   
data_dir="data/train_data"
input_file=[]
gs_file=[]#gold standards
sen1_list=[]
sen2_list=[]
label_list=[]
with open("data/train_data/sicktest")as fi:
    for line in fi:
        line=line.strip().split("\t")
        if len(line)!=3:
            continue
        sen1,sen2,label=line
        sen1_list.append(sen1)
        sen2_list.append(sen2)   
        label_list.append(float(label))
    print("load test file over!!!")
    fi.close()
def main():
    task=sys.argv[1]
    
    if task=="ICLR2017":
        wr_algo(is_ver=False)
    if task=="variety-of-bow":
        sen1_vec,sen2_vec,similarity=my_algo()
    if task=="tf-idf":
        tf_idf()
    if task=="ave-glove-vector":
        wr_algo(is_ver=True)
        
        
if __name__ == "__main__":
    main()            










