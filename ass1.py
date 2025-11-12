import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# --- CNN: Crop Disease Detection ---
X_img, y_img = np.random.rand(100,64,64,3), np.random.randint(2,size=100)
cnn = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(64,64,3)),
    MaxPooling2D(2,2), 
    Flatten(),
    Dense(128,activation='relu'),
    Dense(1,activation='sigmoid')
])
cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
cnn.fit(X_img,y_img,epochs=3,batch_size=10)

# --- Random Forest: Yield Prediction ---
X, y = np.random.rand(100,3), np.random.rand(100)*100
Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.2,random_state=42)
rf = RandomForestRegressor(n_estimators=100,random_state=42).fit(Xtr,ytr)

# --- Recommendation ---
def recommend(d_pred,y_pred):
    if d_pred>=0.5: return "Disease detected! Apply pesticide."
    if y_pred<50: return "Low yield! Improve irrigation."
    return "Crop healthy, yield optimal."

# --- Test with Simulated Inputs ---
test_img = np.random.rand(1,64,64,3)
d_pred = cnn.predict(test_img)[0][0]
y_pred = rf.predict([[0.8,0.6,0.7]])[0]
print(f"Disease Prediction: {d_pred:.4f} (0:Healthy,1:Diseased)")
print(f"Yield Prediction: {y_pred:.2f} units")
print("Recommendation:", recommend(d_pred,y_pred))
