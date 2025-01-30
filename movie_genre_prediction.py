import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
data = {
    'Plot': [
        "Set in medieval India, Queen Padmavati of Mewar is known for her beauty, intelligence, and courage. "
        "She is married to Maharawal Ratan Singh and lives in the Chittor Fort. "
        "The kingdom faces danger when Sultan Alauddin Khilji becomes obsessed with Padmavati and wages war to capture her.",
        
        "A group of superheroes unite to stop a galactic tyrant from obtaining powerful stones that can destroy the universe.",
        "A young wizard named Harry Potter discovers he is famous in a magical world and must battle a dark wizard.",
        "A detective uncovers deep secrets in a seemingly peaceful small town while solving a murder mystery.",
        "Two star-crossed lovers from rival families defy all odds to fight for their love in a hostile world.",
        "In the distant future, humanity sends astronauts through a wormhole to find a new home.",
        "A billionaire inventor builds a high-tech armored suit to fight crime and combat global threats.",
        "A group of teenagers embark on an adventurous quest to find a long-lost pirate treasure.",
        "A scientist experiments with time travel, which leads to unexpected challenges and adventures.",
        "In a dystopian future, a brave young woman sparks a rebellion against an oppressive regime."
    ],
    'Genre': [
        "Historical", "Action", "Fantasy", "Mystery", "Romance", 
        "Sci-Fi", "Action", "Adventure", "Sci-Fi", "Dystopian"
    ]
}


df = pd.DataFrame(data)
vectorizer = TfidfVectorizer(stop_words='english')  # Tfidf for improved text representation
X = vectorizer.fit_transform(df['Plot'])
y = df['Genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")
new_plot = [
    "Set in medieval India, Queen Padmavati of Mewar is known for her beauty, intelligence, and courage. "
    "She is married to Maharawal Ratan Singh and lives in the Chittor Fort. "
    "The kingdom faces danger when Sultan Alauddin Khilji becomes obsessed with Padmavati and wages war to capture her."
]
new_plot_vectorized = vectorizer.transform(new_plot)
predicted_genre = model.predict(new_plot_vectorized)

print(f"The predicted genre for the given plot is: {predicted_genre[0]}")
