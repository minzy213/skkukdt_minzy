# %%
import pandas as pd

# %% [markdown]
# 연도 베스트 셀러
# - 각 연도별 베스트 셀러
# - 유저-best 책
# best 책을 하나만 고르고 그 책의 연도를 가져오는 방법이 있고,     
# 연도별로 평균내서 제일 괜찮은 책을 읽은 연도를 고르는 방법이 있다.
# ---
# 비슷한 유저 추천
# - 유저-책 sparse matrix 만들기
# - knn 혹은 SVD 사용, 구현하기
# ---
# 출판사 베스트 셀러
# - 출판사별 베스트 셀러
# - 유저 best 책
# ---
# 작가 베스트 셀러
# - 작가별 베스트 셀러
# - 유저 best 책
# ---
# 국가 베스트 셀러
# - 국가별 베스트 셀러
# - 유저 국가
# ---
# 매출 높은 책   
# - 가격 * 리뷰수가 가장 높은 책
# ---
# 
# - Country 처리 방법
# - 0으로 평가한 사람 처리 방법(평점 히스토그램 그려보기)
# - 나이로 묶어보고 0 이상 평점 남긴 사람이 각각 많은지 확인해보기. 많으면 나이 써도 괜찮을거같다.
# - 임의로 2명 선택해야 한다.
# -  ```&amp; > &``` 변환 필요
# - 저자, 출판사, 연도 이상한거 없는지 unique 확인
# 
# ---
# 
# 데이터 없거나 평점 남긴게 너무 적은 사람   
# https://www.amazon.com/gp/browse.html?rw_useCurrentProtocol=1&node=8192263011&ref_=bhp_brws_100bks    
# 아마존 홈페이지에 살면서 읽어야 할 책 100권이 있다.    
# 여기에 있는 책중에서 몇개 뽑아서 추천하는 것도 괜찮을 것 같다.

# %% [markdown]
# ## 데이터 가져오기

# %%
columns = ["ISBN"]
df = pd.read_csv('BX-Books.csv', sep = ";", encoding = 'latin', usecols = columns)
columns_to_keep = ["Book-Title", "Book-Author", "Year-Of-Publication", "Publisher"]
book_df = pd.read_csv('BX-Books.csv', sep='";"', encoding = 'latin', usecols = columns_to_keep)
book_df = pd.concat([df, book_df], axis = 1)

user_df = pd.read_csv('BX-Users.csv', sep=';', encoding = 'latin')
rating_df = pd.read_csv('BX-Book-Ratings.csv', sep=';', encoding = 'latin')

# %%
book_df

# %%
user_df

# %%
rating_df

# %% [markdown]
# ## 데이터 정제

# %% [markdown]
# ### Country 추출

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
# #### 국가 없는 사람
# 지역 이름에 국가를 적지 않은 사람 : 4588    
# 첫번째 split도 국가로 했을 때 없는 사람 66

# %%
user_df[user_df['Country'] == '']['Location']

# %% [markdown]
# #### 국가 count 적은 사람
# 
# 국가 count가 2 이하인 사람 : 413
# 
# 첫번째 split도 국가로 했을 때 count가 10 이하인 사람 2590    
# 첫번째 split도 국가로 했을 때 count가 20 이하인 사람 2651    
# 첫번째 split도 국가로 했을 때 count가 30 이하인 사람 2681    
# 첫번째 split도 국가로 했을 때 count가 40 이하인 사람 2703    
# 첫번째 split도 국가로 했을 때 count가 50 이하인 사람 2712    
# 
# ==> 국가 이름을 첫번째 split에 넣은 사람이 많은 것 같다

# %%
out = user_df.groupby('Country').count()[['Location']].reset_index()
out[out['Location'] <= 50]

# %% [markdown]
# ### price parsing

# %%
# Service 지정
import requests
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from tqdm.notebook import tqdm
s = Service(ChromeDriverManager(version="114.0.5735.90").install())
driver = webdriver.Chrome(service=s)
li = []
for id in tqdm(book_df['ISBN']):
    url1='https://www.abebooks.com/book-search/isbn/' + id + '/'
    response = requests.get(url1)
    soup = BeautifulSoup(response.content, 'lxml')
    sel = soup.select('p.item-price')
    if len(sel) > 0:
        li.append(soup.select('p.item-price')[0].text)
    else:
        li.append('')

# %%
lli = pd.DataFrame(li)
lli.to_csv('price.csv')

# %% [markdown]
# ### rating 0 처리

# %% [markdown]
# ### 연도 베스트 셀러
# 
# - 각 연도별 베스트 셀러
# - 유저-best 책
# 
# best 책을 하나만 고르고 그 책의 연도를 가져오는 방법이 있고,     
# 연도별로 평균내서 제일 괜찮은 책을 읽은 연도를 고르는 방법이 있다.

# %% [markdown]
# ### 비슷한 유저 추천
# - 유저-책 sparse matrix 만들기
# - knn 혹은 SVD 사용, 구현하기

# %% [markdown]
# ### 출판사 베스트 셀러
# - 출판사별 베스트 셀러
# - 유저 best 책

# %% [markdown]
# ### 작가 베스트 셀러
# - 작가별 베스트 셀러
# - 유저 best 책

# %% [markdown]
# ### 국가 베스트 셀러
# - 국가별 베스트 셀러
# - 유저 국가

# %% [markdown]
# ### 매출 높은 책
# 가격 * 리뷰수가 가장 높은 책

# %% [markdown]
# ## 대상 5명

# %%
li = [88705,264321,182459,161936,226482]
print(user_df[user_df['User-ID'].isin(li)])

# %% [markdown]
# - 88705 : 1개(0525144609)  
# 
# - 264321 : 850개  
#    53세 user : 2072명 
# 
# - 182459 : 50개  
#    28세 user :  5347명
# 
# - 161936 : 150개    
# 21세 user : 4438명  
# 
# - 226482 : 117개   
# 33세 user : 4700명

# %% [markdown]
# 


