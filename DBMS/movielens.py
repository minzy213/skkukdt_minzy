# %% [markdown]
# # Movielens DB 만들기

# %%
import mysql.connector
import pandas as pd
from tqdm import tqdm

# %%
# db connector 정의
db_conn = mysql.connector.connect(
    host = 'localhost',
    port = 3306,
    user = 'root',
    passwd = '0000'
)
cursor = db_conn.cursor()

# %%
# DB 생성
cursor.execute('CREATE DATABASE ml100k;')
# rating table 생성
cursor.execute('CREATE TABLE ml100k.rating(\
            	user_id INT,\
                movie_id INT,\
                rating INT,\
                ux_tsmp INT\
            )')
db_conn.commit()

# file open
sql = 'INSERT INTO ml100k.rating (user_id, movie_id, rating, ux_tsmp) VALUES (%s, %s, %s, %s);' 
line = []
with open('rating.csv', 'rt') as f:
    for l in f:
        line.append(l.rstrip().split())
    cursor.executemany(sql, line)
    db_conn.commit()
    
cursor.execute("SELECT * FROM ml100k.rating")
myresult = cursor.fetchall()
myresult

# %%
# user table 생성
cursor.execute('CREATE TABLE ml100k.user(\
	user_id INT PRIMARY KEY,\
    age INT,\
    gender CHAR,\
    occupation VARCHAR(20),\
    zip_code VARCHAR(10)\
    );')
db_conn.commit()

# file open
sql = 'INSERT INTO ml100k.user (user_id, age, gender, occupation, zip_code) VALUES (%s, %s, %s, %s, %s);' 
line = []
with open('user.csv', 'rt') as f:
    for l in f:
        line.append(l.rstrip().split('|'))
    cursor.executemany(sql, line)
    db_conn.commit()
    
cursor.execute("SELECT * FROM ml100k.user")
myresult = cursor.fetchall()
myresult

# %%
# movie table 생성
cursor.execute('CREATE TABLE ml100k.movie(\
	movie_id INT PRIMARY KEY,\
    movie_title VARCHAR(255),\
    release_date VARCHAR(20),\
    video_release_date VARCHAR(20),\
    IMDb_URL VARCHAR(255),\
    unknown BOOL,\
    Action BOOL,\
    Adventure BOOL,\
    Animation BOOL,\
    Children BOOL,\
    Comedy BOOL,\
    Crime BOOL,\
    Documentary BOOL,\
    Drama BOOL,\
    Fantasy BOOL,\
    FilmNoir BOOL,\
    Horror BOOL,\
    Musical BOOL,\
    Mystery BOOL,\
    Romance BOOL,\
    SciFi BOOL,\
    Thriller BOOL,\
    War BOOL,\
    Western  BOOL\
);')
db_conn.commit()

# file open
sql = 'INSERT INTO ml100k.movie VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);' 
line = []
with open('movie2.csv', 'rt') as f:
    ii = 0
    for l in f:
        line.append(l.rstrip().split('|')) 
        ii += 1
        if(ii % 5000 == 0):
            line = []
            cursor.executemany(sql, line)
            db_conn.commit()
cursor.executemany(sql, line)
db_conn.commit()

cursor.execute("SELECT * FROM ml100k.movie")
myresult = cursor.fetchall()
myresult

# %%
# 외래키 설정
cursor.execute('ALTER TABLE ml100k.rating ADD FOREIGN KEY(user_id) REFERENCES ml100k.user(user_id) ON DELETE CASCADE;')
cursor.execute('ALTER TABLE ml100k.rating ADD FOREIGN KEY(movie_id) REFERENCES ml100k.movie(movie_id) ON DELETE CASCADE;')
db_conn.commit()

# %%
movie_df = pd.read_csv('movie.csv', sep = '|', encoding='latin1', header = None)
movie_df.columns = ['movie_id' , 'movie_title' , 'release_date' , 'video_release_date' ,
              'IMDb_URL' , 'unknown' , 'Action' , 'Adventure' , 'Animation' ,
              'Children' , 'Comedy' , 'Crime' , 'Documentary' , 'Drama' , 'Fantasy' ,
              'Film-Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci-Fi' ,
              'Thriller' , 'War' , 'Western']

# %%
movie_df

# %%
monthMap = {'Jan' : 1, 'Feb' : 2, 'Mar' : 3, 'Apr' : 4, 'May' : 5, 'Jun' : 6, 'Jul' : 7, 'Aug' : 8, 'Sep' : 9, 'Oct' : 10, 'Nov' : 11, 'Dec' : 12}
def getYear(data):
    s = data.split('-')
    if len(s) > 1:
        return s[2]
    else:
        print
movie_df['rlsYear'] = movie_df['release_date'].apply(lambda x : str(x).split('-')[2])

# %%

movie_df['rlsYear'] = movie_df['release_date'].apply(lambda x : str(x).split('-')[2])
movie_df['rlsMonth'] = movie_df['release_date'].apply(lambda x : str(x).split('-')[1])
movie_df['rlsMonth'] = movie_df['rlsMonth'].map(monthMap)
movie_df['rlsDay'] = movie_df['release_date'].apply(lambda x : str(x).split('-')[0])

# %%


# %%
movie_df.to_csv('movie2.csv', index = False)

# %%
cursor.execute('ALTER TABLE ml100k.rating DROP FOREIGN KEY rating_ibfk_2;')
cursor.execute('DROP TABLE ml100k.movie;')
db_conn.commit()

# %%



