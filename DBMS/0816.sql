CREATE DATABASE testDB;
USE testDB;
CREATE TABLE Persons ( 
	PersonID int, 
	LastName varchar(255), 
	FirstName varchar(255), 
	Address varchar(255), 
	City varchar(255) 
);

ALTER TABLE Persons
ADD email varchar(64),
MODIFY COLUMN City varchar(32);

ALTER TABLE world.city
ADD new_col varchar(64);

-- use sakila
USE sakila;
-- 가장 길이가 긴 영화의 제목은?
SELECT title
FROM film
ORDER BY length DESC
LIMIT 1;		-- 가장 길이가 긴 영화가 여러개다. 모르겠다.

-- 장편 영화(100분 이상) 들의 영화 장르를 모두 출력해 보세요
SELECT DISTINCT(name) FROM category WHERE category_id IN
(SELECT category_id FROM film_category WHERE film_id IN 
(SELECT film_id FROM film WHERE length >= 100));

-- 드라마 장르 영화의 설명 중 love가 들어가는 영화는 어떤 것이 있나요?
SELECT title FROM film WHERE description LIKE '%love%' and film_id in
(SELECT film_id FROM film_category WHERE category_id IN
(SELECT category_id FROM category WHERE name = 'Drama'));  -- 없음!

-- 'Killer Innocent' 라는 영화에 출연한 배우들의 first, last name
SELECT first_name, last_name FROM actor WHERE actor_id IN
(SELECT actor_id FROM film_actor WHERE film_id IN
(SELECT film_id FROM film WHERE title = 'KILLER INNOCENT'));

-- 영화 제목에 'wood'가 들어간 영화는?
SELECT title FROM film WHERE title LIKE '%WOOD%';

-- 'Christmas Moonshine'을 빌려간 고객 명단
SELECT first_name, last_name FROM customer WHERE customer_id IN
(SELECT customer_id FROM rental WHERE inventory_id IN
(SELECT inventory_id FROM inventory WHERE film_id IN
(SELECT film_id FROM film WHERE title = 'CHRISTMAS MOONSHINE')));