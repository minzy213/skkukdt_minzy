# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
# 파일 읽어오기
rating_df = pd.read_csv('u.data', sep='\t', header=None, names = ['userID', 'movieID', 'rating', 'timestamp'])

# %%
# Reader = 읽기 위한 객체, 이걸로 rating_df를 dataset으로 읽어온다.
from surprise import Dataset, Reader

rdr = Reader(line_format="user item rating", sep="\t") ## 반드시 사용자-아이템-평점 순서로
data = Dataset.load_from_df(rating_df[['userID', 'movieID', 'rating']], reader=rdr)

# %%
# split
from surprise.model_selection import train_test_split
trainset, testset = train_test_split(data, test_size=0.2, random_state=1004)
testset

# %% [markdown]
# ## KNN으로 추천시스템 맛보기

# %%
# KNN 모델 불러와서 fit
from surprise import KNNBasic
recom_sys = KNNBasic()
recom_sys.fit(trainset)

# %%
# testset의 예측 결과 보기
# uid=741, iid=281, r_ui=2.0, est=3.0157043977599276
# userid, itemid, actual, predict
pred = recom_sys.test(testset)
pred[:10]

# %%
# testset의 rmse error 확인하기
from surprise import accuracy
accuracy.rmse(recom_sys.test(testset))

# %%
# 예측 결과 예쁘게 출력해보기
for p in pred[:20]:
    print(f'user : {p.iid:<4}, movie : {p.uid}, rating : {p.r_ui} -- > {p.est:.2f}, diff : {p.r_ui - p.est:>5.2f}')

# %%
# error 그래프 그려보기
error_list = []
for p in pred:
    error_list.append(p.r_ui - p.est)
    
plt.style.use('seaborn')
plt.hist(error_list, bins = 20)
# vertical line. 이 안에 들어오는 것들은 반올림하면 정확하게 예측된 것.
plt.axvline(-0.5, color = 'red')
plt.axvline(0.5, color = 'red')
# 이 안에 들어오는 것들은 반올림하면 별점 1 틀린 것.
plt.axvline(-1.5, color = 'orange')
plt.axvline(1.5, color = 'orange')

# %% [markdown]
# 별점 예측은 지도 학습, 회귀 모델.

# %% [markdown]
# ### 랜덤으로 testdata 하나 만들어서 이사람한테 영화 추천해주기!
# 
# 1. 유저 선택   
# 1. 유저가 안본 영화 리스트   
# 1. 영화 리스트에 대한 평점 전부 예측   
# 1. 예측 평점이 가장 높은 N개의 영화 선택

# %%
# 1. 유저 선택
import random
# 랜덤으로 testdata 하나 만들어서 사람 하나 뽑기
usr, mv, rating = testset[random.randint(0, 20000)]
usr, mv, rating

# %%
movie_info_df = pd.read_csv('u.item', sep='|', encoding='latin', header=None)
movie_info_df.columns = ['movieID' , 'movie_title' , 'release_date' , 'video_release_date' ,
                        'IMDb_URL' , 'unknown' , 'Action' , 'Adventure' , 'Animation' ,
                        'Children' , 'Comedy' , 'Crime' , 'Documentary' , 'Drama' , 'Fantasy' ,
                        'Film-Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci-Fi' ,
                        'Thriller' , 'War' , 'Western']

movie_name = movie_info_df.loc[mv]
print(movie_name['movie_title'], '- predicted score : ', pred.est, 'real score :', rating)
print(f'{movie_name.movie_title} - predicted score : , {pred.est:.2f}, real score : {rating}')

# %%
# 2. 유저가 안본 영화 리스트

# 전체 영화 id 가져옴
all_movie_ids = rating_df.movieID.unique()
# 이 사람이 본 영화 id 가져옴
watched_movie_ids = rating_df[rating_df['userID']==usr].movieID
# 전체 영화에서 이 사람이 본 영화 id 삭제
target_mv_list = set(all_movie_ids) - set(watched_movie_ids)

# %%
# 영화 리스트에 대한 평점 전부 예측, 
# 하나의 유저에 대해 모든 target mv list의 예상 점수 dict
# dataframe으로 바로 변환하려면 dictionary를 list로 저장!
all_pred_scores = []
for s_mv_id in target_mv_list:
    
    all_pred_scores.append({'movieID' : s_mv_id, 'pred_score':recom_sys.predict(usr, s_mv_id).est})

pred_df = pd.DataFrame(all_pred_scores)
pred_df

# %%
# 예측 평점이 가장 높은 N개의 영화 선택
pred_df = pred_df.sort_values('pred_score', ascending=False)
recom_mv_ids = pred_df[pred_df['pred_score'] == 5]

# %%
result_df = pd.merge(movie_info_df, recom_mv_ids, on='movieID')
result_df['movie_title']

# %% [markdown]
# ### 많은 모델의 예측 성능 보기

# %%
from surprise import KNNBasic, BaselineOnly, CoClustering, KNNBaseline, NMF, SVD
models = [KNNBasic(), BaselineOnly(), CoClustering(random_state=0), KNNBaseline(), NMF(random_state=0), SVD(random_state=0)]

for m in models:
    m.fit(trainset)
    accuracy.rmse(m.test(testset))


# %%



