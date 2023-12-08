library(stats)  ## for dist
#install.packages("NbClust")
library(NbClust)
library(cluster)
library(mclust)
library(amap)  ## for Kmeans (notice the cap K)
library(factoextra) ## for cluster vis, silhouette, etc.
library(MAP)
library(purrr)
#install.packages("stylo")
library(stylo)  ## for dist.cosine
#install.packages("philentropy")
library(philentropy)  ## for distance() which offers 46 metrics
library(SnowballC)
library(caTools)
library(dplyr)
library(textstem)
library(stringr)
library(wordcloud)
library(tm) ## to read in corpus (text data)
setwd("E:/GU/501/data")
Labeledcsvdata<-read.csv("Cleaned_Diabetes_Dataset.csv")
## Save the label
Label_csv <- Labeledcsvdata$Outcome
## Remove the label from the dataset
## remove column 9
csvdata <- Labeledcsvdata[ ,-c(9)]
head(csvdata)
### Look at the pairwise distances between the vectors (rows, points)
(Dist1<- dist(csvdata, method = "minkowski", p=1)) ##Manhattan
(Dist2<- dist(csvdata, method = "minkowski", p=2)) #Euclidean
(DistE<- dist(csvdata, method = "euclidean")) #same as p = 2
## Create a normalized version of Record_3D_DF
(csvdata_Norm <- as.data.frame(apply(csvdata[,1:3 ], 2, ##2 for col
                                          function(x) (x - min(x))/(max(x)-min(x)))))
## Look at scaled distances
(Dist_norm<- dist(csvdata_Norm, method = "minkowski", p=2)) #Euclidean
## use scale in R
csvdata_scale<-scale(csvdata)
## NbClust helps to determine the number of clusters.
kmeans_1<-NbClust::NbClust(csvdata_Norm, 
                              min.nc=2, max.nc=5, method="kmeans")
## How many clusters is best....let's SEE.........
table(kmeans_1$Best.n[1,])

barplot(table(kmeans_1$Best.n[1,]), 
        xlab="Numer of Clusters", ylab="",
        main="Number of Clusters")

## Does Silhouette agree?
fviz_nbclust(csvdata_Norm, method = "silhouette", 
             FUN = hcut, k.max = 5)
## Two clusters (k = 2) is likely the best. 
# Silhouette Coefficient = (x-y)/ max(x,y)
# y: mean in cluster
# x: mean dist from nearest cluster
# -1 means val in wrong cluster
# 1 means right cluster

## Elbow Method (WSS - within sum sq)
############################# Elbow Methods ###################

fviz_nbclust(
  as.matrix(csvdata_Norm), 
  kmeans, 
  k.max = 5,
  method = "wss",
  diss = get_dist(as.matrix(csvdata_Norm), method = "manhattan")
)
## k means..............
######################################
kmeans_1_Result <- kmeans(csvdata, 2, nstart=25)   
## I could have used the normalized data - which is better to use
## But - by using the non-norm data, the results make more visual
## sense - which also matters.

# Print the results
print(kmeans_1_Result)
summary(kmeans_1_Result)

kmeans_1_Result$centers  

aggregate(csvdata, 
          by=list(cluster=kmeans_1_Result$cluster), mean)
## Compare to the labels
table(Labeledcsvdata$Label, kmeans_1_Result$cluster)
## This is a confusion matrix with 100% prediction (very rare :)
summary(kmeans_1_Result)
## Place results in a tbale with the original data
cbind(Labeledcsvdata, cluster = kmeans_1_Result$cluster)
## See each cluster
kmeans_1_Result$cluster
# Cluster size
kmeans_1_Result$size
## Visualize the clusters
fviz_cluster(kmeans_1_Result, csvdata, main="Spearman")
## k = 2
My_Kmeans_2<-Kmeans(csvdata_Norm, centers=2 ,method = "spearman")
fviz_cluster(My_Kmeans_2, csvdata, main="Spearman")
## k= 3
My_Kmeans_3<-Kmeans(csvdata_Norm, centers=3 ,method = "spearman")
fviz_cluster(My_Kmeans_3,csvdata, main="Spearman")
## k = 2 with Euclidean
My_Kmeans_E2<-Kmeans(csvdata_Norm, centers=2 ,method = "Euclidean")
fviz_cluster(My_Kmeans_E2, csvdata, main="Euclidean")
## k = 3 with Euclidean
My_Kmeans_E3<-Kmeans(csvdata_Norm, centers=3 ,method = "euclidean")
fviz_cluster(My_Kmeans_E3, csvdata, main="Euclidean")
## Heat maps...
## Recall that we have Dist2..
Dist2<- dist(csvdata_Norm, method = "minkowski", p=2) #Euclidean
fviz_dist(Dist2, gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))+ggtitle("Euclidean Heatmap")
#######################################################
## 
##          Hierarchical CLustering
Dist_norm_M2<- dist(csvdata_Norm, method = "minkowski", p=2) #Euclidean
## Now run hclust...you may use many methods - Ward, Ward.D2, complete, etc..
## see above
HClust_Ward_Euc_N_3D <- hclust(Dist_norm_M2, method = "average" )
plot(HClust_Ward_Euc_N_3D, cex=0.9, hang=-1, main = "Minkowski p=2 (Euclidean)")
rect.hclust(HClust_Ward_Euc_N_3D, k=4)

dist_mat <- dist(csvdata, method = 'euclidean')
hclust_avg <- hclust(dist_mat, method = 'average')
plot(hclust_avg,main="Hierarchical Clustering")
suppressPackageStartupMessages(library(dendextend))
avg_dend_obj <- as.dendrogram(hclust_avg)
avg_col_dend <- color_branches(avg_dend_obj, h = 4)
plot(avg_col_dend)
## Using Man with Ward.D2..............................
dist_C <- stats::dist(csvdata_Norm, method="manhattan")
HClust_Ward_CosSim_N_3D <- hclust(dist_C, method="ward.D2")
plot(HClust_Ward_CosSim_N_3D, cex=.7, hang=-30,main = "Manhattan")
rect.hclust(HClust_Ward_CosSim_N_3D, k=2)

vec1<-c(132,87,43,125,33.8,0.389,45,3)
vec2<-c(23,34,44,134,55.8,0.789,67,2)
vec3<-c(234,45,33,155,44.8,0.987,34,0)
centers<-kmeans_1_Result$centers
dist1<-dist(rbind(vec1,vec2,vec3,centers), method = "euclidean")  
dist1
