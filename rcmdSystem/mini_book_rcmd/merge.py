# %%
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors

import warnings
warnings.filterwarnings("ignore")

# %%
# 데이터 read
columns = ["ISBN"]
df = pd.read_csv('BX-Books.csv', sep = ";", encoding = 'latin', usecols = columns)
columns_to_keep = ["Book-Title", "Book-Author", "Year-Of-Publication", "Publisher"]
book_df = pd.read_csv('BX-Books.csv', sep='";"', encoding = 'latin', usecols = columns_to_keep)
book_df = pd.concat([df, book_df], axis = 1)

user_df = pd.read_csv('BX-Users.csv', sep=';', encoding = 'latin')
rating_df = pd.read_csv('BX-Book-Ratings.csv', sep=';', encoding = 'latin')
user = [88705, 264321, 182459, 161936, 226482, 11676, 198711]


# %%
li = [88705,264321,182459,161936,226482, 11676, 198711]
for i in li:
    print(user_df[user_df['User-ID'] == i].values)

# %% [markdown]
#  ## 현준님

# %%
#############################################데이터 전처리 과정#######################################################
# ISBN 결측치 삭제
rating_df = rating_df[rating_df['ISBN'].isin(book_df['ISBN'])]
# User df 에서 나라 정보 분리
Location_country = []
for country in user_df['Location']:
    for i in range(len(country)):
        if country[:: - 1][i] == ',':
            Location_country.append(country[:: - 1][0:i][::-1])
            ### split(',')[-1]
            break

# 공백 제거
real_Location_country = []
for i in Location_country:
    real_Location_country.append(i.strip()) ### replace(' ', '')

# 200번에 'united kingdom' 빠져있음 ㅡㅡ 추가
real_Location_country.insert(200, 'united kingdom')

# user_df의 2번째 열에 country 추가 및 Location 열 제거 
user_df.insert(1, 'Country', real_Location_country)
user_df.drop('Location', axis = 1, inplace = True)

# User별 거주국가 카운팅
Users_by_Contry = pd.DataFrame(data = user_df['Country'].value_counts(), index = user_df['Country'].unique())
Users_by_Contry = Users_by_Contry.sort_values('Country', ascending = False)

# user가 가장 많이 거주하는 나라 top_8과 그 외 나라에서 거주하는 user
Users_by_Contry_Top_8 = Users_by_Contry.head(8)
Users_by_Contry_Top_8.rename(index = {"" : 'other'}, inplace = True)

# 나라 이름 첫 글자 대문자 변경, Usa 전부 대문자 변경
top_7 =  ['usa', 'canada', 'united kingdom', 'germany', 'spain', 'australia', 'italy']
contry_name = []
for i in user_df['Country']:
    if i in top_7:
        contry_name.append(i.title())
    else:
        contry_name.append('Other')

def ch_str(x):
    if x == 'Usa':
        x = "USA"
    return x
user_df['Country'] = user_df['Country'].apply(ch_str) ### string.upper()

# user_df data에 user별 거주 지역 data 병합
user_df.insert(1, 'User_Contry', contry_name)
user_df.drop('Country', axis = 1, inplace = True)



# %%
# 외부에서 user 추천
# 해당유저가 읽은 책 제외
# new_df에서 아래 대로 진행하고 user의 거주지역에서 bestseller n권

def best_seller_recommand_country(b_df, target_user_idx, recommand_num):
    retBooks = []
    recommand_df = pd.merge(b_df, user_df)
    user_country = user_df[user_df['User-ID'] == target_user_idx]['User_Contry'].values[0] # 해당 유저의 거주국가 
    # 해당 유저의 국가의 best n 추천
    User_Country_Best_Seller = recommand_df[recommand_df['User_Contry'] == user_country]['ISBN'].value_counts()[:recommand_num].index
    retBooks.extend(book_df[book_df['ISBN'].isin(User_Country_Best_Seller)]['ISBN'].tolist())
    
    return retBooks


# %% [markdown]
#  ## 윤지님

# %%
# 평점 1이상이면서, 리뷰 10개 이상인 책들의 평균평점이 가장 높은 책 추천
def best_seller_recommand_rating(b_df, recommand_num):
    nonzero_rating = b_df[b_df['Book-Rating'] >= 1] #평점 1점 이상인 데이터들만 저장
    nonzero_sort = nonzero_rating.groupby('ISBN').count().sort_values('Book-Rating', ascending = False)
    review_over10 = nonzero_sort[nonzero_sort['User-ID'] >= 10].reset_index() #평점 1이상이면서, 리뷰 10개 이상인 책
    review_over10 = review_over10.drop('User-ID', axis = 1) #유저아이디 제거
    review_over10.rename(columns = {'Book-Rating' : 'Rating-Count'}, inplace = True) #column명 변경
    #평점 1이상이면서, 리뷰 10개 이상인 책들의 평균평점 구하기
    total_rating = pd.merge(nonzero_rating, review_over10, on = 'ISBN', how = 'inner')
    total_recom = total_rating.groupby('ISBN').mean().sort_values('Book-Rating', ascending = False).reset_index()
    return total_recom.iloc[:recommand_num]['ISBN'].tolist()



# %% [markdown]
#  ## 상혁님

# %%
# 가장 많이 팔린 책 추천
# best seller rating 0 포함
def best_seller_recommand_selling(b_df, recommand_num):
    book_sold = b_df.groupby('ISBN').count().sort_values('Book-Rating', ascending = False)[['Book-Rating']]
    # 책 best seller rating 0 포함 - 읽었던 책은 뺌
    book_sold.reset_index(inplace = True)
    return book_sold[:recommand_num]['ISBN'].tolist()
    


# %%
# KNN 대규모 추천
def fitKnn():
    user_ids = rating_df['User-ID'].unique()
    user2idx_dict = {x:i for i, x in enumerate(user_ids)}

    book_ids = rating_df['ISBN'].unique()
    book2idx_dict = {x:i for i, x in enumerate(book_ids)}

    rating_df['u_idx'] = rating_df['User-ID'].map(user2idx_dict)
    rating_df['b_idx'] = rating_df['ISBN'].map(book2idx_dict)

    num_users = rating_df['u_idx'].nunique()
    num_books = rating_df['b_idx'].nunique()

    sparse_arr = coo_matrix((rating_df['Book-Rating'], (rating_df['u_idx'], rating_df['b_idx'])), shape = (num_users, num_books))
    # 끝

    # k가 850일 때 최적
    knn = NearestNeighbors(n_neighbors = 850, metric = 'cosine')
    knn.fit(sparse_arr)
    return knn

def best_seller_recommand_knn(curknn, target_user_idx):
    target_rating_df = rating_df[rating_df['User-ID'] == target_user_idx]

    num_books = rating_df['b_idx'].nunique()
    target_sparse_arr = coo_matrix((target_rating_df['Book-Rating'], ([0]*len(target_rating_df['u_idx']), target_rating_df['b_idx'])), shape = (1, num_books))

    dist, idx = curknn.kneighbors(target_sparse_arr, n_neighbors = 850)
    neighbors_df = rating_df[rating_df['u_idx'].isin(idx[0][1:])]

    pred_df = neighbors_df.groupby('ISBN').mean()['Book-Rating'].sort_values(ascending = False)
    pred_df = pred_df.reset_index()
    pred_df.columns = ['ISBN', 'pred']

    result_df = pd.merge(target_rating_df, pred_df, on = 'ISBN')[['ISBN', 'Book-Rating', 'pred', 'u_idx']]

    return result_df.groupby('ISBN').mean().sort_values('pred', ascending= False).reset_index()['ISBN'][:3].values.tolist()
        


# %% [markdown]
#  ## 석호님
# - 유저가 가장 좋아하는 작가의 책 추천
# - 가장 평점이 좋은 책, 가장 많이 팔린 책, 가장 많이 팔렸으면서 평점이 좋은 책
# - 평점을 남긴 책이 10권 이상인 유저만 작가 베이스 추천   
# ---
# - 264321 best author : Joe Haldeman
# - 182459 best author : J. K. Rowling
# - 161936 best author : Dave Pelzer
# - 226482 best author : Jane Green
# - 11676 best author : Alice Sebold
# - 198711 best author : Marguerite S. Herman

# %%
# 유저가 가장 좋아하는 작가의 책 추천
# 가장 좋아하는 작가 : 평점*리뷰수
# 가장 평점이 좋은 책, 가장 많이 팔린 책, 가장 많이 팔렸으면서 평점이 좋은 책
def best_seller_recommand_author(b_df, target_user_idx, recommand_num):
    # author_merged_df
    author_rating_df = b_df.merge(book_df[['ISBN','Book-Author']], on='ISBN')
    author_rating_df.head()
    # 평점 col 생성
    book_mean_df = book_df.merge(b_df[['ISBN','Book-Rating']].groupby('ISBN').mean(),on='ISBN')
    book_mean_df=book_mean_df.rename(columns = {'Book-Rating':'mean_rating'})
    # 리뷰수 col 생성
    book_mean_df = book_mean_df.merge(b_df[['ISBN','Book-Rating']].groupby('ISBN').count(),on='ISBN')
    book_mean_df=book_mean_df.rename(columns = {'Book-Rating':'count_rating'})
    # 평점*리뷰수 col 생성
    book_mean_df['mean_count'] = book_mean_df.mean_rating * book_mean_df.count_rating

    #ISBN, max-score
    isbn_target = b_df[b_df['User-ID'] == target_user_idx]['ISBN']
    max_score = b_df[b_df['User-ID'] == target_user_idx]['Book-Rating'].max()
    # best author
    best_author_list = author_rating_df[(author_rating_df['User-ID'] == target_user_idx) & (author_rating_df['Book-Rating'] == max_score)].groupby('Book-Author').count().sort_values('User-ID',ascending=False).index
    li = []
    if len(best_author_list) > 0:
        best_author = book_mean_df[book_mean_df['Book-Author'].isin(best_author_list)].sort_values('mean_count',ascending=False).iloc[0]['Book-Author']
        # by mean
        target_books = book_mean_df[(book_mean_df['Book-Author'] == best_author)& ~(book_mean_df.ISBN.isin(isbn_target))]
        li.extend(target_books.sort_values(by=['mean_count','mean_rating'],ascending=False)['ISBN'][:recommand_num].tolist())
    return li


# %%
fittedKnn = fitKnn()
users = [88705, 264321, 182459, 161936, 226482, 11676, 198711]
for cur_user in users:
    li, author, knn = [], [], []
    readBooks = rating_df[(rating_df['User-ID'] == cur_user) & (rating_df['Book-Rating'] > 0)].count()['User-ID']
    # 유저가 읽은 책 제외한 dataframe
    user_book_df = rating_df[rating_df['User-ID'] != cur_user]
    if readBooks >= 10:
        author = best_seller_recommand_author(rating_df, cur_user, 3) # 읽었던 책이 필요해서 rating_df
        knn = best_seller_recommand_knn(fittedKnn, cur_user) # 3권
        country = best_seller_recommand_country(user_book_df, cur_user, 4)
        rating = best_seller_recommand_rating(user_book_df, 4)
        selling = best_seller_recommand_selling(user_book_df, 4)
    else:
        country = best_seller_recommand_country(user_book_df, cur_user, 5)
        rating = best_seller_recommand_rating(user_book_df, 5)
        selling = best_seller_recommand_selling(user_book_df, 5)
        
    # 우선이 되는 지표에 가중치 부여
    li = author*5 + knn*3 + country*2 + rating + selling
    rcmdbook = pd.DataFrame(li, columns = ['ISBN'])
    rcmdbook['count'] = 0
    rcmdbook = rcmdbook.groupby('ISBN').count().reset_index().sort_values('count', ascending = False)
    rcmdbook = pd.merge(rcmdbook, book_df[['ISBN', 'Book-Title']], on = 'ISBN', how = 'left')
    print(f'user : {cur_user}')
    for title in rcmdbook['Book-Title'][:10]:
        print(title)
    print('='*15)


# %%



