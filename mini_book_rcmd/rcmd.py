# %%
import pandas as pd


# %% [markdown]
#  연도 베스트 셀러
#  - 각 연도별 베스트 셀러
#  - 유저-best 책
#  best 책을 하나만 고르고 그 책의 연도를 가져오는 방법이 있고,
#  연도별로 평균내서 제일 괜찮은 책을 읽은 연도를 고르는 방법이 있다.
#  ---
#  비슷한 유저 추천
#  - 유저-책 sparse matrix 만들기
#  - knn 혹은 SVD 사용, 구현하기
#  ---
#  출판사 베스트 셀러
#  - 출판사별 베스트 셀러
#  - 유저 best 책
#  ---
#  작가 베스트 셀러
#  - 작가별 베스트 셀러
#  - 유저 best 책
#  ---
#  국가 베스트 셀러
#  - 국가별 베스트 셀러
#  - 유저 국가
#  ---
#  매출 높은 책
#  - 가격 * 리뷰수가 가장 높은 책
#  - 가격에 변동이 있을 수 있고, 다 가져오는거는 시간이 너무 오래 걸린다.    
#   필요한 책 몇개의 가격만 가져오는게 시간상 옳다.
#  ---
#  - Country 처리 방법
#   지역 이름에 국가를 적지 않은 사람 : 4588   
#   국가 count가 2 이하인 사람 : 413    
#    첫번째 split도 국가로 했을 때 없는 사람 66    
#   첫번째 split도 국가로 했을 때 count가 10 이하인 사람 2590     
#   첫번째 split도 국가로 했을 때 count가 20 이하인 사람 2651     
#   첫번째 split도 국가로 했을 때 count가 30 이하인 사람 2681     
#   첫번째 split도 국가로 했을 때 count가 40 이하인 사람 2703     
#   첫번째 split도 국가로 했을 때 count가 50 이하인 사람 2712     
# 
#  - 0으로 평가한 사람 처리 방법(평점 히스토그램 그려보기)    
#     리뷰 0 뺀 갯수 433671, 유저 : 77805명   
#     리뷰 전체 갯수 1149780, 유저 : 105283명
# 
#  - 나이로 묶어보고 0 이상 평점 남긴 사람이 각각 많은지 확인해보기. 많으면 나이 써도 괜찮을거같다.    
#    전체 리뷰 : 1149780     
#    나이가 존재하는 리뷰 : 305277    
#    8<= 나이 <= 80인 리뷰 : 302285   
# 
#  - 임의로 2명 선택해야 한다.
# 
#  -  ```&amp; > &``` 변환 필요
#  
#  - 저자, 출판사, 연도 이상한거 없는지 unique 확인
# 
#  ---
#  
#  데이터 없거나 평점 남긴게 너무 적은 사람     
#  https://www.amazon.com/gp/browse.html?rw_useCurrentProtocol=1&node=8192263011&ref_=bhp_brws_100bks     
#  아마존 홈페이지에 살면서 읽어야 할 책 100권이 있다.     
#  여기에 있는 책중에서 몇개 뽑아서 추천하는 것도 괜찮을 것 같다.

# %% [markdown]
#  ## 데이터 가져오기

# %%
columns = ["ISBN"]
df = pd.read_csv('BX-Books.csv', sep = ";", encoding = 'latin', usecols = columns)
columns_to_keep = ["Book-Title", "Book-Author", "Year-Of-Publication", "Publisher"]
book_df = pd.read_csv('BX-Books.csv', sep='";"', encoding = 'latin', usecols = columns_to_keep)
book_df = pd.concat([df, book_df], axis = 1)

user_df = pd.read_csv('BX-Users.csv', sep=';', encoding = 'latin')
rating_df = pd.read_csv('BX-Book-Ratings.csv', sep=';', encoding = 'latin')


# %%
rating_df = rating_df.merge(book_df, on='ISBN')
rating_df = rating_df[['User-ID', 'ISBN', 'Book-Rating']]
rating_df.head()

# %%
print(len(book_df.Publisher))
book_df.groupby('Publisher').count().sort_values('ISBN').reset_index()[['Publisher', 'ISBN']]

# %%
user_df


# %%
rating_df


# %% [markdown]
#  ## 데이터 정제

# %% [markdown]
#  ### Country 추출

# %%
# Location에서 국가 정보 추출하기
def getCountry(data):
    s = data.replace(' ', '').replace('"', '').split(',')
    if len(s) > 0:
        if len(s[-1]):
            return s[-1]
        else: # 마지막 split에 국가 없을 때 첫번째 split에서 가져와본다
            return s[0]
user_df['Country'] = user_df['Location'].apply(getCountry)


# %% [markdown]
#  #### 국가 없는 사람
#  지역 이름에 국가를 적지 않은 사람 : 4588    
#  첫번째 split도 국가로 했을 때 없는 사람 66

# %%
user_df[user_df['Country'] == '']['Location']


# %% [markdown]
#  #### 국가 count 적은 사람
# 
#  국가 count가 2 이하인 사람 : 413
# 
#  첫번째 split도 국가로 했을 때 count가 10 이하인 사람 2590     
#  첫번째 split도 국가로 했을 때 count가 20 이하인 사람 2651     
#  첫번째 split도 국가로 했을 때 count가 30 이하인 사람 2681     
#  첫번째 split도 국가로 했을 때 count가 40 이하인 사람 2703     
#  첫번째 split도 국가로 했을 때 count가 50 이하인 사람 2712     
# 
#  ==> 국가 이름을 첫번째 split에 넣은 사람이 많은 것 같다

# %%
out = user_df.groupby('Country').count()[['Location']].reset_index()
out[out['Location'] <= 50]

# %%
user_df['Country'].nunique()

# %% [markdown]
#  ### price parsing

# %%
# Service 지정
# import requests
# from webdriver_manager.chrome import ChromeDriverManager
# from bs4 import BeautifulSoup
# from tqdm.notebook import tqdm
# s = Service(ChromeDriverManager(version="114.0.5735.90").install())
# driver = webdriver.Chrome(service=s)
# li = []
# for id in tqdm(book_df['ISBN']):
#     url1='https://www.abebooks.com/book-search/isbn/' + id + '/'
#     response = requests.get(url1)
#     soup = BeautifulSoup(response.content, 'lxml')
#     sel = soup.select('p.item-price')
#     if len(sel) > 0:
#         li.append(soup.select('p.item-price')[0].text)
#     else:
#         li.append('')


# %%
# lli = pd.DataFrame(li)
# lli.to_csv('price.csv')


# %% [markdown]
#  ### rating 0 처리

# %% [markdown]
#  ### 연도 베스트 셀러
# 
#  - 각 연도별 베스트 셀러
#  - 유저-best 책
# 
#  best 책을 하나만 고르고 그 책의 연도를 가져오는 방법이 있고,
#  연도별로 평균내서 제일 괜찮은 책을 읽은 연도를 고르는 방법이 있다.

# %% [markdown]
#  ### 비슷한 유저 추천
#  - 유저-책 sparse matrix 만들기
#  - knn 혹은 SVD 사용, 구현하기

# %% [markdown]
#  ### 출판사 베스트 셀러
#  - 출판사별 베스트 셀러
#  - 유저 best 책

# %% [markdown]
#  ### 작가 베스트 셀러
#  - 작가별 베스트 셀러
#  - 유저 best 책

# %% [markdown]
#  ### 국가 베스트 셀러
#  - 국가별 베스트 셀러
#  - 유저 국가

# %% [markdown]
#  ### 매출 높은 책
#  가격 * 리뷰수가 가장 높은 책

# %% [markdown]
#  ## 대상 5명

# %%
li = [88705,264321,182459,161936,226482, 11676, 198711]
print(user_df[user_df['User-ID'].isin(li)])


# %%
user_df[user_df['User-ID'] == 11676]

# %% [markdown]
#  - 88705 : 1개(0525144609)
# 
#  - 264321 : 850개
#     53세 user : 2072명
# 
#  - 182459 : 50개
#     28세 user :  5347명
# 
#  - 161936 : 150개
#  21세 user : 4438명
# 
#  - 226482 : 117개
#  33세 user : 4700명

# %%
rating_df[rating_df['Book-Rating'] > 0]['Book-Rating'].hist(bins = 10)

# %%
# 평점 히스토그램
rating_df['Book-Rating'].hist()

# %%
nonZero = rating_df[rating_df['Book-Rating'] != 0]
hasAge = user_df[user_df['Age'] >= 0][['User-ID', 'Age']]
hasAge

# %%
nonZeroAge = nonZero.merge(hasAge, on = 'User-ID')
nonZeroAge.groupby('Age').mean()

# %%
nonZeroAge = nonZeroAge[(8 <= nonZeroAge['Age']) & (nonZeroAge['Age'] <= 80)]
nonZeroAge.groupby('Age').count()[['User-ID']].to_csv('tmp.csv')

# %% [markdown]
# # SVD

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix # array도 있음

# %%
rating_df

# %%
# linalg.svds 연산할 때 float여야 한다고 한다.
rating_df['Book-Rating'] = rating_df['Book-Rating'].apply(float)
rating_df

# %%
rating_df['User-ID'].unique()

# %%
# sparse matrix 만들때는 행, 열 id가 모두 0부터 시작하는 연속적인 숫자여야 한다.
# 연속적인 숫자로 만들어야 하는데, 그냥 무작위로 만들어버리면 원본 데이터와 연결이 안되므로
# mapping table을 미리 만들어 둔다.
u_id = rating_df['User-ID'].unique()
user2idx = {org : new for new, org in enumerate(u_id)}
b_id = rating_df['ISBN'].unique()
book2idx = {org : new for new, org in enumerate(b_id)}


rating_df['u_idx'] = rating_df['User-ID'].map(user2idx) # apply는 복잡한거 할 때, map은 category 단순 매핑할때 사용
rating_df['b_idx'] = rating_df['ISBN'].map(book2idx)
rating_df

# %%
# sparse matrix 만들 때 전체 shape이 어떻게 되는지 지정해야 한다.
# 전체 행, 열 갯수 지정하려고 nunique 사용-
num_user = rating_df['u_idx'].nunique()
num_book = rating_df['b_idx'].nunique()

# %%
# train, valid, test split : 8:1:1
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(rating_df, test_size = 0.5, random_state = 1004)

# %%
# sparse matrix 만들기, csr
from scipy.sparse import csr_matrix
sparse_df = coo_matrix((train_df['Book-Rating'], (train_df['u_idx'], train_df['b_idx'])), 
                shape = (num_user, num_book))

# %%
# 행렬 분해, U, S, V는 ndarray
from scipy.sparse import linalg
U, S, V = linalg.svds(sparse_df, k = 30)

# %%
# target 유저의 영화 선호도 예측 정보 가져온다.
import warnings
warnings.filterwarnings("ignore")
# 11676
target_user_idx = 20
for kk in range(50, 101, 10):
    U, S, V = linalg.svds(sparse_df, k = kk, random_state = 100)
    
    err = []
    for i in range(0, num_user, 1000):
        pred = U[i] @ np.diag(S) @ V

        # target_df 만들기
        target_rated_df = train_df[train_df['u_idx'] == i]
        target_rated_df['pred'] = pred[target_rated_df['b_idx'].to_list()]*10**17

        err.append(target_rated_df[['Book-Rating']].corrwith(target_rated_df['pred']).values[0])
    print(np.mean(err))

# %%
for i in range(1, 9):
    U, S, V = linalg.svds(sparse_df, k = i, random_state = 100)

    err = []
    pred = U[i] @ np.diag(S) @ V

    # target_df 만들기
    target_rated_df = train_df[train_df['u_idx'] == user2idx[264321]]
    target_rated_df['pred'] = pred[target_rated_df['b_idx'].to_list()]

    target_rated_df['Book-Rating'] = target_rated_df[['Book-Rating']]
    print(i, target_rated_df[['Book-Rating']].corrwith(target_rated_df['pred']).values[0])

# %%
U, S, V = linalg.svds(sparse_df, k = 2000, random_state = 100)

err = []
pred = U[i] @ np.diag(S) @ V +

# target_df 만들기
target_rated_df = train_df[train_df['u_idx'] == user2idx[264321]]
target_rated_df['pred'] = pred[target_rated_df['b_idx'].to_list()]

target_rated_df['Book-Rating'] = target_rated_df[['Book-Rating']]
print(i, target_rated_df[['Book-Rating']].corrwith(target_rated_df['pred']).values[0])

# %%
# rating 평균 상위 top5
nonzero_rating = rating_df[rating_df['Book-Rating'] >= 1] #평점 1점 이상인 데이터들만 저장
nonzero_sort = nonzero_rating.groupby('ISBN').count().sort_values('Book-Rating', ascending = False)
review_over10 = nonzero_sort[nonzero_sort['User-ID'] >= 10].reset_index() #평점 1이상이면서, 리뷰 10개 이상인 책
review_over10 = review_over10.drop('User-ID', axis = 1) #유저아이디 제거
review_over10.rename(columns = {'Book-Rating' : 'Rating-Count'}, inplace = True) #column명 변경
#평점 1이상이면서, 리뷰 10개 이상인 책들의 평균평점 구하기
total_rating = pd.merge(nonzero_rating, review_over10, on = 'ISBN', how = 'inner')
total_recom = total_rating.groupby('ISBN').mean().sort_values('Book-Rating', ascending = False).reset_index()
print('평균 평점이 가장 높은 Top 5')
for i in range(5): #상위 5개 추출
    print(f'{i+1}번째 추천', total_recom.iloc[i]['ISBN'])

# %%
# 베스트셀러
for target_user_idx in [88705, 264321, 182459, 161936, 226482, 11676, 198711]:
    # best seller rating 0 포함
    book_sold = rating_df.groupby('ISBN').count().sort_values('Book-Rating', ascending = False)[['Book-Rating']]
    book_sold
    # 유저가 읽은 책을 추천 책에서 빼야 한다.
    # target_user_idx = 264321
    target_user_read_df = rating_df[rating_df['User-ID'] == target_user_idx]
    target_user_read_df
    # 책 best seller rating 0 포함 - 읽었던 책은 뺌
    book_sold.reset_index(inplace = True)
    target_notread_best_seller_df = book_sold[~book_sold['ISBN'].isin(target_user_read_df['ISBN'])] # best seller 중 target 유저가 안 읽은 책들
    top3_target_notread_best_seller_df = target_notread_best_seller_df[:5]
    print(f'{target_user_idx}님의 책 추천 top3 =')
    print(top3_target_notread_best_seller_df, end ='\n'*2)


