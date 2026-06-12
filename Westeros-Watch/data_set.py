import pandas as pd
data = {
    "character":      ["Tywin", "Cersei", "Ned", "Jon", "Tyrion", "Joffrey", "Daenerys", "Sansa",
                       "Ramsay", "Arya", "Jaime", "Littlefinger", "Varys", "The Hound", "Brienne",
                       "Stannis", "Theon", "Margaery", "Bronn", "Samwell", "Hodor"],
    "betrayals":      [3, 5, 0, 2, 2, 2, 2, 2, 1, 0, 1, 4, 3, 2, 0, 1, 2, 1, 1, 0, 0],
    "kills_ordered":  [5, 8, 1, 1, 2, 2, 10, 2, 6, 8, 3, 4, 0, 6, 3, 3, 2, 0, 4, 1, 0],
    "allies":         [10, 3, 9, 9, 5, 4, 8, 5, 4, 3, 6, 6, 6, 2, 3, 3, 3, 6, 4, 3, 3],
    "power_rank":     [10, 9, 5, 7, 7, 6, 9, 5, 1, 2, 6, 8, 7, 1, 3, 9, 3, 6, 1, 0, 0],
    "weapon_strength":[3, 3, 9, 10, 2, 2, 8, 1, 7, 9, 9, 0, 0, 9, 8, 8, 7, 0, 7, 3, 0],
    "deception":      [9, 8, 0, 3, 7, 2, 2, 5, 1, 7, 1, 10, 8, 1, 0, 0, 2, 6, 0, 0, 0],
    "threat":         ["High", "High","Low","Medium","Medium","High","High","Low","High","High", "Medium", "High", "High", "Medium", "Medium", "Medium", "Medium", "Medium", "Medium", "Low","Low"],
}
df = pd.DataFrame(data)
df
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X = df[["betrayals", "kills_ordered", "allies", "power_rank", "weapon_strength", "deception"]]   # double brackets = list of columns
y = df["threat"]     
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)   # convention: truth first, guesses second
print(accuracy)
