use testdb;
CREATE TABLE movie(
	movie_id INT PRIMARY KEY,
    movie_title VARCHAR(255),
    release_date VARCHAR(20),
    video_release_date VARCHAR(20),
    IMDb_URL VARCHAR(255),
    unknown BOOL,
    Action BOOL,
    Adventure BOOL,
    Animation BOOL,
    Children BOOL,
    Comedy BOOL,
    Crime BOOL,
    Documentary BOOL,
    Drama BOOL,
    Fantasy BOOL,
    FilmNoir BOOL,
    Horror BOOL,
    Musical BOOL,
    Mystery BOOL,
    Romance BOOL,
    SciFi BOOL,
    Thriller BOOL,
    War BOOL,
    Western  BOOL
);

CREATE TABLE user(
	user_id INT PRIMARY KEY,
    age INT,
    gender CHAR,
    occupation VARCHAR(20),
    zip_code VARCHAR(10)
);

CREATE TABLE rating(
	user_id INT,
    movie_id INT,
    rating INT,
    timestamp INT
);
SET foreign_key_checks = 1;
ALTER TABLE rating ADD FOREIGN KEY(user_id) REFERENCES user(user_id) ON DELETE CASCADE;
ALTER TABLE rating ADD FOREIGN KEY(movie_id) REFERENCES movie(movie_id) ON DELETE CASCADE;

SELECT * FROM movie;

-- 무비렌즈를 DB로
-- 가장 최근에 개봉된 영화의 평균 평점

ALTER TABLE movie
ADD year int,
ADD month int,
ADD day int;
INSERT INTO movie (year, month, day)
-- 평가를 가장 많이한 회원이 본 영화의 리스트를 평점 순으로
-- 학생의 숫자
-- 의사들이 가장 좋아한 영화