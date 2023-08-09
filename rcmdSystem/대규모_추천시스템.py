# %% [markdown]
# ## Sparse Matrix 데이터 구조

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix # array도 있음

# %%
# 좌표, 값, 전체 행렬의 크기

data = np.array([1, 5, 3, 4, 2, 6])
row = np.array([0, 1, 0, 1, 0, 2])
col = np.array([0, 1, 2, 0, 1, 2])

coo = coo_matrix((data, (row, col)), shape = (10, 10))
print('--coo 형식-----')
print(coo.todense())

csr = csr_matrix((data, (row, col)), shape=(10, 10))
print('--csr 형식-----')
print(csr.todense())

csc = csc_matrix((data, (row, col)), shape=(10, 10))
print('--csc 형식-----')
print(csc.todense())

# %%
# 행 기반 합산
print("COO 행 기반 합산:\n", coo.sum(axis=1))
print("CSR 행 기반 합산:\n", csr.sum(axis=1))
print("CSC 행 기반 합산:\n", csc.sum(axis=1))

# 열 기반 합산
print("CSR 열 기반 합산:\n", csr.sum(axis=0))
print("CSC 열 기반 합산:\n", csc.sum(axis=0))

# %% [markdown]
# ### sparse, dense 비교

# %%
data = np.ones(50000)
row = np.random.randint(0, 10000, size = 50000) # 임의의 50000개 좌표 생성
col = np.random.randint(0, 10000, size = 50000)


coo = coo_matrix((data, (row, col)), shape=(10000, 10000))
csr = coo.tocsr()
csc = coo.tocsc()

arr = csr.toarray()

# %%
# 크기 비교
print("Sparse Matrix (COO) 크기:", coo.data.nbytes + coo.row.nbytes + coo.col.nbytes)
print("Scipy sparse matrix (CSR) 크기:", csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes)
print("Scipy sparse matrix (CSC) 크기:", csc.data.nbytes + csc.indices.nbytes + csc.indptr.nbytes)
print("Numpy toarray (CSR) 크기:", arr.nbytes)

# %%
# 속도 비교
from time import time

t = time()
coo.sum(axis=1)
print(time() - t)

t = time()
csr.sum(axis=1)
print(time() - t)

t = time()
csc.sum(axis=1)
print(time() - t)


# %% [markdown]
# ### Movie data 실습

# %%

rating_df = pd.read_csv('..\dataset\movie.data', sep='\t', header=None, names = ['userID', 'movieID', 'rating', 'timestamp'])[['userID', 'movieID', 'rating']]
rating_df

# %%
# pivot table로 바꾸면 na가 아주 많은 null이 있다
pv_df=pd.pivot_table(data=rating_df, values='rating',index='movieID',columns='userID')
pv_df.fillna('-')

# %%
# user 방향 sparse matrix 만들기
# id가 1부터 할당되어 있어서 갯수가 안맞다. 모두 -1 해줘야 한다.
ml_spm = csr_matrix((rating_df['rating'], (rating_df['userID']-1, rating_df['movieID']-1)),
                           shape=(rating_df['userID'].nunique(), rating_df['movieID'].nunique()))
ml_spm

# %%
rating_df['movieID'].nunique()

# %%
ml_dense = ml_spm.toarray()
ml_dense

# %% [markdown]
# ## KNN-sklearn으로 대규모 추천시스템 만들어보기

# %%
# 데이터 불러오기
rating_df = pd.read_csv('..\dataset\movie.data', sep='\t', header=None, names = ['userID', 'movieID', 'rating', 'timestamp'])[['userID', 'movieID', 'rating']]
movie_info_df = pd.read_csv('..\dataset\movie.item', sep='|', encoding='latin')
movie_info_df.columns = ['movieID' , 'movie_title' , 'release_date' , 'video_release_date' ,
              'IMDb_URL' , 'unknown' , 'Action' , 'Adventure' , 'Animation' ,
              'Children' , 'Comedy' , 'Crime' , 'Documentary' , 'Drama' , 'Fantasy' ,
              'Film-Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci-Fi' ,
              'Thriller' , 'War' , 'Western']

# %% [markdown]
# ### 데이터 준비

# %% [markdown]
# #### id -> index
# - 0번부터 시작
# - 중간에 비어 있는게 없어야 한다.
# 

# %%
# mapping table
# {id: index, ...}
user_ids = rating_df['userID'].unique()
user2idx_dict = {x : i for i, x in enumerate(user_ids)}
idx2user_dict = {i : x for i, x in enumerate(user_ids)}

movie_ids = rating_df['movieID'].unique()
movie2idx_dict = {x : i for i, x in enumerate(movie_ids)}
idx2movie_dict = {i : x for i, x in enumerate(movie_ids)}

rating_df['u_idx'] = rating_df['userID'].map(user2idx_dict) # apply는 복잡한거 할 때, map은 category 단순 매핑할때 사용
rating_df['i_idx'] = rating_df['movieID'].map(movie2idx_dict) # apply는 복잡한거 할 때, map은 category 단순 매핑할때 사용

# %%
rating_df.sort_values(['u_idx', 'i_idx'])

# %% [markdown]
# #### 추천시스템을 위한 데이터의 분포 조사
# 
# 영화별로 몇 명이나 봤는지 조사

# %%
rating_df.groupby('movieID').count()['userID'].sort_values(ascending = False)

# %%
movies_count_df = pd.DataFrame(rating_df.groupby('movieID').size(), columns=['count'])
movies_count_df.head()
ax = movies_count_df.sort_values('count', ascending=False) \
                    .reset_index(drop=True) \
                    .plot(
                        figsize=(8, 6),
                        title='Rating Frequency of All Movies',
                        fontsize=12
                    )
ax.set_xlabel("movie Id")
ax.set_ylabel("number of ratings")

# %% [markdown]
# #### sparse matrix 만들기
# data, row_index, co_index, shape(row, col)

# %%
num_user = rating_df['u_idx'].nunique()
num_movie = rating_df['i_idx'].nunique()

# %% [markdown]
# #### train, val, test data split
# 
# 6:2:2, 6:3:1, 7:2:1, 7:1.5:1.5 ....   
# 이번에는 9:0.5:0.5. colaboration filtering은 train data양이 중요하다.

# %%
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(rating_df, test_size = 0.1, random_state = 1004)
val_df, test_df = train_test_split(test_df, test_size = 0.5, random_state = 1004)

# %%
from scipy.sparse import coo_matrix
train_sp = coo_matrix((train_df['rating'], 
                 (train_df['u_idx'], train_df['i_idx'])), 
                shape = (num_user, num_movie))

# %% [markdown]
# ### 모델 빌드 및 학습

# %%
knn = NearestNeighbors(n_neighbors=50, metric = 'cosine')
knn.fit(train_sp)

# %% [markdown]
# ### 결과 추론 및 평가

# %%
# target user를 sp matrix 형태로 선언
target_user_idx = 8
target_tr_df = train_df[train_df['u_idx'] == target_user_idx]
data = [4, 5, 4, 4]
a = [0, 0, 0, 0]
b = [209, 170, 935, 355]
target_sp = coo_matrix((target_tr_df['rating'], ([0] * len(target_tr_df), target_tr_df['i_idx'])),
           shape = (1, num_movie))

# %%
len(target_tr_df['i_idx'])

# %%
dist, idx = knn.kneighbors(target_sp, n_neighbors = 50)

# %%
idx

# %%
# 8번 유저에 대한 이웃들의 영화 평점 예측
n_df = rating_df[rating_df['u_idx'].isin(idx[0][1:])]
pred_df = n_df.groupby('movieID').mean()['rating'].sort_values(ascending = False)
pred_df = pred_df.reset_index().rename({'rating':'pred'}, axis = 1)
pred_df

# %%
# 평가

target_te_df = test_df[test_df['u_idx'] == target_user_idx]
target_te_df = target_te_df.sort_values('rating')
target_te_df

# %%
pred_te_df = pred_df[pred_df['movieID'].isin(target_te_df['movieID'])].sort_values('pred')
pred_te_df

# %% [markdown]
# - 평점은 조금 틀려보인다.(RMSE)
# - 추천시스템은 top N개를 추천할 때 그게 맞는지가 중요함.
# - 순서를 잘 맞추냐 : nDCG

# %%
resul_df = pd.merge(target_te_df, pred_te_df, on = 'movieID', how='inner')[['movieID', 'rating', 'pred']]
resul_df

# %% [markdown]
# ### 모든 유저로 일반화

# %%
# target user를 sp matrix 형태로 선언

target_user_idx = 8
total_prd_df = pd.DataFrame(data = None, columns = ['movieID', 'userID', 'rating', 'pred'])
for target_user_idx in range(num_user):
    target_tr_df = train_df[train_df['u_idx'] == target_user_idx]
    target_sp = coo_matrix((target_tr_df['rating'], ([0] * len(target_tr_df), target_tr_df['i_idx'])),
            shape = (1, num_movie))
    dist, idx = knn.kneighbors(target_sp, n_neighbors = 50)

    # 8번 유저에 대한 이웃들의 영화 평점 예측
    n_df = rating_df[rating_df['u_idx'].isin(idx[0][1:])]
    pred_df = n_df.groupby('movieID').mean()['rating'].sort_values(ascending = False)
    pred_df = pred_df.reset_index().rename({'rating':'pred'}, axis = 1)

    # 평가
    target_te_df = test_df[test_df['u_idx'] == target_user_idx]
    target_te_df = target_te_df.sort_values('rating')

    pred_te_df = pred_df[pred_df['movieID'].isin(target_te_df['movieID'])]

    resul_df = pd.merge(target_te_df, pred_te_df, on = 'movieID', how='inner')[['movieID', 'userID', 'rating', 'pred']]
    total_prd_df = pd.concat([total_prd_df, resul_df])
total_prd_df

# %%
total_prd_df.head(20)

# %%
import plotly.express as px
px.violin(total_prd_df, x='rating', y='pred', box=True)

# %%
from sklearn.metrics import mean_squared_error
mean_squared_error(total_prd_df['rating'], total_prd_df['pred'], squared = False)

# %%
# 점수별 RMSE
for i in range(1, 6):
    rmse = mean_squared_error(total_prd_df[total_prd_df['rating'] == i]['rating'], 
                              total_prd_df[total_prd_df['rating'] == i]['pred'], squared = False)
    print(i, rmse)

# %% [markdown]
# #### 분류 관점에서 좋아할 영화/아닌 영화로 나눠보자

# %%
good_df = total_prd_df[total_prd_df['rating'] > 3]
bad_df = total_prd_df[total_prd_df['rating'] < 3]

# %%
good_df['target_bool'] = 1
bad_df['target_bool'] = 0
cls_df = pd.concat([good_df, bad_df])
cls_df

# %%
from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_predictions(
    cls_df['target_bool'], # real 평점
    (cls_df['pred'] - 1)/4, # pred 확률
    name=f"knn ",
    color="darkorange",
)

plt.plot([0, 1], [0, 1], color='k', label='Random Model', linestyle = '--', alpha = 0.6)

plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC AUC")
plt.legend()
plt.show()

# %%
# 복습하고 최적의 k값을 찾아보기
# 115 0.9902222162421708
# 71  0.9931089248584817

# target user를 sp matrix 형태로 선언
for i in range(20, 80, 10):
    knn = NearestNeighbors(n_neighbors=i, metric = 'cosine')
    knn.fit(train_sp)

    total_prd_df = pd.DataFrame(data = None, columns = ['movieID', 'userID', 'rating', 'pred'])
    for target_user_idx in range(num_user):
        target_tr_df = train_df[train_df['u_idx'] == target_user_idx]
        target_sp = coo_matrix((target_tr_df['rating'], ([0] * len(target_tr_df), target_tr_df['i_idx'])),
                shape = (1, num_movie))
        dist, idx = knn.kneighbors(target_sp, n_neighbors = i)

        # 8번 유저에 대한 이웃들의 영화 평점 예측
        n_df = rating_df[rating_df['u_idx'].isin(idx[0][1:])]
        pred_df = n_df.groupby('movieID').mean()['rating'].sort_values(ascending = False)
        pred_df = pred_df.reset_index().rename({'rating':'pred'}, axis = 1)

        # 평가
        target_vl_df = val_df[val_df['u_idx'] == target_user_idx]
        target_vl_df = target_vl_df.sort_values('rating')

        pred_te_df = pred_df[pred_df['movieID'].isin(target_vl_df['movieID'])]

        resul_df = pd.merge(target_vl_df, pred_te_df, on = 'movieID', how='inner')[['movieID', 'userID', 'rating', 'pred']]
        total_prd_df = pd.concat([total_prd_df, resul_df])
    total_prd_df

    rmse = mean_squared_error(total_prd_df['rating'], total_prd_df['pred'], squared = False)
    print(i, rmse)

# %%



