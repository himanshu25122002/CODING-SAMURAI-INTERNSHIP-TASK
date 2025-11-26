# ----- Import Libraries -----
import pandas as pd
import matplotlib.pyplot as plt

# ----- Load Dataset -----
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# ----- Merge movie & rating data -----
df = pd.merge(ratings, movies, on="movieId")

# ----- Analyze Most Popular Movies Based on Average Ratings -----
movie_ratings = df.groupby("title")["rating"].mean().sort_values(ascending=False)
top_movies = movie_ratings.head(10)
print("🔹 Top 10 Movies Based on Average Ratings:")
print(top_movies)

# -----  Group Movies by Genre and Visualize Count of Movies per Genre -----
genre_split = movies["genres"].str.split("|").explode()
genre_count = genre_split.value_counts()

plt.figure(figsize=(10,6))
genre_count.plot(kind="bar")
plt.title("Number of Movies per Genre")
plt.xlabel("Genre")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# -----  Recommend Top-Rated Movies by User-Selected Genre -----
def recommend_by_genre(genre, top_n=10):
    # Filter movies in selected genre
    genre_movies = movies[movies["genres"].str.contains(genre, case=False, na=False)]
    merged = pd.merge(genre_movies, ratings, on="movieId")
    
    avg_rated = merged.groupby("title")["rating"].mean().sort_values(ascending=False)
    return avg_rated.head(top_n)

# Test recommendation function
user_genre = input("\nEnter a genre to get movie recommendations (e.g., Action, Drama, Comedy): ")
recommended = recommend_by_genre(user_genre)
print(f"\n Top movies in genre '{user_genre}':")
print(recommended)
