# %% [markdown]
# ## DB 접근, 읽어오기

# %%
import mysql.connector
import pandas as pd
from tqdm import tqdm

# %% [markdown]
# ![image.png](attachment:image.png)

# %%
# db connector 정의
mydb = mysql.connector.connect(
    host = 'localhost',
    port = 3306,
    user = 'root',
    passwd = '0000',
    database = 'world'
)

# %%
# 커서 저장
cur = mydb.cursor()

# %%
# 쿼리문 서버로 보냄
cur.execute('SELECT * FROM city')

# %%
# 쿼리 결과 전부 가져옴(fetch)
qryResult = cur.fetchall()
qryResult

# %%
city_df = pd.DataFrame(qryResult, columns = ['ID', 'Name', 'CountryCode', 'District', 'Population', 'new_col'])
city_df

# %%
from matplotlib import pyplot
city_df.Population.hist(bins = 100)

# %%
# 새로운 데이터 추가
cur.execute('INSERT INTO city (ID, Name, CountryCode) VALUES (0, "Seoul", "KOR")')
# select와 다르게 insert, delete, update 등은 commit 필요.
mydb.commit()

# %%
# 추가한 데이터(서울) 확인, 서울이 두개다!
cur.execute("SELECT * FROM city WHERE Name = 'Seoul'")
myresult = cur.fetchall()

for x in myresult:
  print(x)

# %%
# 인구가 0인 서울(내가 위에서 추가한 서울) 삭제
cur.execute("DELETE FROM city WHERE CountryCode = 'KOR' AND Population = 0")
mydb.commit()
cur.execute("SELECT * FROM city WHERE Name = 'Seoul'")
myresult = cur.fetchall()

for x in myresult:
  print(x)

# %% [markdown]
# ## 데이터 추가하기

# %% [markdown]
# ### 하나씩 추가할 때

# %%
val = (0, 'Paradise', 'KOR')
# 문자열 추가할때는 " " 꼭 씌워서 넣어야 한다!!!
cur.execute('INSERT INTO city (ID, Name, CountryCode) VALUES ({}, "{}", "{}");'.format(val[0], val[1], val[2]))
mydb.commit()

# %% [markdown]
# ### 여러개 추가할 때

# %%
val = [(0, 'Idea', 'KOR'), (0, 'Utopia', 'KOR')]
for v in val:
    cur.execute('INSERT INTO city (ID, Name, CountryCode) VALUES ({}, "{}", "{}");'.format(*v))
    # *v ==> 자동으로 묶여서 [0][1][2] 들어간다.
mydb.commit()

# %%
cur.execute("SELECT * FROM city WHERE CountryCode = 'KOR' AND Population = 0")
myresult = cur.fetchall()

for x in myresult:
  print(x)

# %%
qry = 'INSERT INTO city (ID, Name, CountryCode) VALUES (%s, %s, %s);' # 숫자도 %s로.
val = [(0, 'Paradise', 'KOR'), (0, 'Utopia', 'KOR'), (0, 'Idea', 'KOR')]
cur.executemany(qry, val)
mydb.commit()

# %%
cur.execute("SELECT * FROM city WHERE CountryCode = 'KOR' AND Population = 0")
myresult = cur.fetchall()

for x in myresult:
  print(x)

# %% [markdown]
# ## Pandas를 사용해 DB 처리

# %%
# 읽어오기
movie_df = pd.read_sql_query('SELECT * FROM testdb.movie', mydb)
movie_df

# %%
monthMap = {'Jan' : 1, 'Feb' : 2, 'Mar' : 3, 'Apr' : 4, 'May' : 5, 'Jun' : 6, 'Jul' : 7, 'Aug' : 8, 'Sep' : 9, 'Oct' : 10, 'Nov' : 11, 'Dec' : 12}
def getYear(data):
    s = data.split('-')
    if len(s) > 1:
        return s[2]
    else:
        print
movie_df = movie_df.drop(['video_release_date', 'year', 'month', 'day'], axis = 1)
movie_df = movie_df.dropna()
movie_df['rlsYear'] = movie_df['release_date'].apply(lambda x : str(x).split('-')[2])
movie_df['rlsMonth'] = movie_df['release_date'].apply(lambda x : str(x).split('-')[1])
movie_df['rlsMonth'] = movie_df['rlsMonth'].map(monthMap)
movie_df['rlsDay'] = movie_df['release_date'].apply(lambda x : str(x).split('-')[0])

# %% [markdown]
# ## pandas > mysql 데이터 쓰기
# 

# %%
from sqlalchemy import create_engine, types
import pymysql
import mysql.connector
import pandas as pd
from tqdm import tqdm

# %%
movie_df = pd.read_csv('movie.csv', sep = '|', encoding='latin1', header = None)
movie_df.columns = ['movie_id' , 'movie_title' , 'release_date' , 'video_release_date' ,
              'IMDb_URL' , 'unknown' , 'Action' , 'Adventure' , 'Animation' ,
              'Children' , 'Comedy' , 'Crime' , 'Documentary' , 'Drama' , 'Fantasy' ,
              'Film-Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci-Fi' ,
              'Thriller' , 'War' , 'Western']

monthMap = {'Jan' : 1, 'Feb' : 2, 'Mar' : 3, 'Apr' : 4, 'May' : 5, 'Jun' : 6, 'Jul' : 7, 'Aug' : 8, 'Sep' : 9, 'Oct' : 10, 'Nov' : 11, 'Dec' : 12}
def getYear(data):
    s = data.split('-')
    if len(s) > 1:
        return s[2]
    else:
        print
movie_df = movie_df.drop(['video_release_date'], axis = 1)
movie_df = movie_df.dropna()
movie_df['rlsYear'] = movie_df['release_date'].apply(lambda x : str(x).split('-')[2])
movie_df['rlsMonth'] = movie_df['release_date'].apply(lambda x : str(x).split('-')[1])
movie_df['rlsMonth'] = movie_df['rlsMonth'].map(monthMap)
movie_df['rlsDay'] = movie_df['release_date'].apply(lambda x : str(x).split('-')[0])

# %%
movie_df

# %%
db_connection_str = f'mysql+pymysql://root:0000@localhost/ml100k'
db_conn = create_engine(db_connection_str)
ml100k_conn = db_conn.connect()

# %%
movie_df.to_sql('movie', con = ml100k_conn, if_exists='replace')

# %%
#https://docs.sqlalchemy.org/en/20/core/type_basics.html
dtypesql = {'movie_id':types.Integer(),
          'movie_title':types.String(255),
          'release_date':types.String(20),
          'IMDb_URL':types.String(255),
          'unknown':types.Boolean(),
          'Action':types.Boolean(),
          'Adventure':types.Boolean(),
          'Animation':types.Boolean(),
          'Children':types.Boolean(),
          'Comedy':types.Boolean(),
          'Crime':types.Boolean(),
          'Documentary':types.Boolean(),
          'Drama':types.Boolean(),
          'Fantasy':types.Boolean(),
          'Film-Noir':types.Boolean(),
          'Horror':types.Boolean(),
          'Musical':types.Boolean(),
          'Mystery':types.Boolean(),
          'Romance':types.Boolean(),
          'Sci-Fi':types.Boolean(),
          'Thriller':types.Boolean(),
          'War':types.Boolean(),
          'Western':types.Boolean(),
          'rlsYear':types.String(20),
          'rlsMonth':types.String(20),
          'rlsDay':types.String(20)
}
movie_df.to_sql(name='movie', con=ml100k_conn, if_exists='replace', dtype=dtypesql)

# %%
movie_df.columns

# %%



