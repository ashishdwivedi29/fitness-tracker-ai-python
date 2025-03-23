import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings("ignore")
import openai
import os
from dotenv import load_dotenv

# Load datasets
exercise_df = pd.read_csv("exercise.csv")
calories_df = pd.read_csv("calories.csv")
disease_df = pd.read_csv("disease_data.csv")  # Dataset for disease prediction


# Merge datasets on User_ID
df = pd.merge(exercise_df, calories_df, on="User_ID")
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

# Preprocess data for calorie prediction
X = df.drop(columns=["User_ID", "Calories"])
y = df["Calories"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Section 1: Algorithm Description
# Algorithm selection
st.title("Personal Fitness Tracking with AI")
algo_info = {
    "Random Forest": "An ensemble learning method using decision trees to improve accuracy.",
    "XGBoost": "An optimized gradient boosting algorithm with high predictive power.",
    "Neural Network": "A deep learning model that mimics human brain neurons for complex data.",
}

model_choice = st.sidebar.selectbox("Choose Model", list(algo_info.keys()))
st.write(f"### {model_choice} Algorithm")
st.write(algo_info[model_choice])

# Train Model
if model_choice == "XGBoost":
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X_train, y_train)
elif model_choice == "Neural Network":
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

# Section 2: User Information and Calorie Prediction
st.header("Calorie Burn Estimation")
user_data = {
    "Gender": st.sidebar.selectbox("Gender", ["Male", "Female"]),
    "Age": st.sidebar.number_input("Age", min_value=10, max_value=100, value=25),
    "Height": st.sidebar.number_input("Height (cm)", min_value=100, max_value=220, value=175),
    "Weight": st.sidebar.number_input("Weight (kg)", min_value=30, max_value=150, value=70),
    "Duration": st.sidebar.number_input("Workout Duration (min)", min_value=5, max_value=180, value=30),
    "Heart_Rate": st.sidebar.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=120),
    "Body_Temp": st.sidebar.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0)
}
user_data["Gender"] = 0 if user_data["Gender"] == "Male" else 1
input_df = pd.DataFrame([user_data])

prediction = model.predict(input_df)
calories_per_min = prediction[0] / user_data["Duration"]
st.write(f"Estimated Calories Burned: **{round(prediction[0], 2)} kcal**")
st.write(f"Calories per min: **{round(calories_per_min, 2)} kcal/min**")
for t in [30, 60, 90]:
    st.write(f"For {t} min: **{round(calories_per_min * t, 2)} kcal**")

# Section 3: Progress Tracking with Calendar
st.header("Daily, Weekly, Monthly Goals & Progress Tracker")

# Create a dataframe to track history
date_today = datetime.date.today()
dates = pd.date_range(start=date_today - datetime.timedelta(days=30), periods=31).date
goal_status = {date: False for date in dates}  # Default: Not Achieved

selected_goal = st.selectbox("Select Goal Type", ["Daily", "Weekly", "Monthly"])
selected_date = st.date_input("Select Date", min_value=dates[0], max_value=dates[-1])
completed = st.checkbox("Mark Goal as Completed")

# Update goal status
goal_status[selected_date] = completed

# Display goal tracking history
st.subheader("Goal Tracking History")
history_df = pd.DataFrame({"Date": list(goal_status.keys()), "Achieved": list(goal_status.values())})
st.dataframe(history_df)

# Display Calendar-like Completion Status
st.subheader("Goal Completion Calendar")
df_calendar = history_df.copy()
df_calendar["Achieved"] = df_calendar["Achieved"].replace({True: "✅", False: "❌"})
st.write(df_calendar.set_index("Date"))

# Monthly Progress Chart
df_chart = history_df.copy()
df_chart["Calories Burned"] = np.random.randint(200, 600, size=len(df_chart))  # Simulated Data
df_chart.loc[df_chart["Date"] == date_today, "Calories Burned"] = prediction[0]
st.line_chart(df_chart.set_index("Date"))

# SECTION 4: AI Chatbot for Fitness Assistance
# Load API key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Client
client = openai.OpenAI(api_key=api_key)

def get_fitness_advice(user_query):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a fitness expert."},
                {"role": "user", "content": user_query}
            ]
        )
        return response.choices[0].message.content  # Return AI's response
    except Exception as e:
        return f"Error: {e}"

#some in-built responses 
st.header("\U0001F4AC AI Fitness Chatbot")

responses = {
    "hello": "Hello! How can I assist you with your fitness journey?",
    "how are you": "I'm just a bot, but I'm here to help with your fitness queries!",
    "calories": "Tracking calories is important! Try to maintain a balance based on your daily activities.",
    "exercise": "Exercise daily for at least 30 minutes for a healthy lifestyle.",
    "bye": "Goodbye! Stay fit and take care! \U0001F60A"
}

user_input = st.text_input("You:", "")

if st.button("Send"):
    user_input = user_input.lower().strip()
    if user_input in responses:
        response = responses[user_input]
    else:
        response = get_fitness_advice(user_input)  # Get AI-generated response
    
    st.text_area("Chatbot:", response, height=100)


# Section 5: Shopping Cart
st.header("Fitness Store - Add to Cart")
products = {"Protein Powder": 20, "Dumbbells": 30, "Yoga Mat": 15}
cart = []
for product, price in products.items():
    if st.checkbox(f"Add {product} - ${price}"):
        cart.append((product, price))

# Add clickable links
product_links = {"Protein Powder": "https://example.com/protein", "Dumbbells": "https://example.com/dumbbells", "Yoga Mat": "https://example.com/yogamat"}
st.write("### Product Links")
for product, link in product_links.items():
    st.markdown(f"[{product}]({link})")

total = sum([item[1] for item in cart])
st.write(f"Total Cart Value: **${total}**")

# SECTION 6: Disease Risk Prediction with Improved Model
st.header("Disease Risk Prediction")

disease_features = disease_df.drop(columns=["Disease"])
disease_labels = disease_df["Disease"].astype('category').cat.codes  # Convert categorical labels to numeric

disease_X_train, disease_X_test, disease_y_train, disease_y_test = train_test_split(
    disease_features, disease_labels, test_size=0.2, random_state=42
)

disease_model = RandomForestClassifier(n_estimators=100, random_state=42)
disease_model.fit(disease_X_train, disease_y_train)

user_disease_input = np.random.rand(1, disease_features.shape[1])  # Placeholder user input
disease_pred_prob = disease_model.predict_proba(user_disease_input)[0]

diseases = disease_df["Disease"].unique()
for i, disease in enumerate(diseases):
    risk_percentage = round(disease_pred_prob[i] * 100, 2)
    st.write(f"**{disease} Risk:** {risk_percentage}%")
    st.progress(disease_pred_prob[i])


@st.cache_data
def load_data():
    df = pd.read_csv("exercise_dataset.csv")  # Use the uploaded dataset
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
    return df

df = load_data()

#section 6A: visualization
st.header("📊 Visualization of User Data")
st.subheader("📊 Calories Burned vs. Exercise Type (Bar Chart)")
if "Exercise" in df.columns and "Calories Burn" in df.columns:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=df["Exercise"], y=df["Calories Burn"], ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlabel("Exercise Type")
    ax.set_ylabel("Calories Burned")
    ax.set_title("Calories Burned for Different Exercise Types")
    st.pyplot(fig)
    st.markdown("📌 **Pros:** Clearly shows calorie differences for each exercise.\n")
    st.markdown("📌 **Cons:** Doesn't show variations within each type.")
else:
    st.error("⚠️ Required columns not found!")

st.subheader("❤️ Age vs. Heart Rate (Scatter Plot)")
if "Age" in df.columns and "Heart Rate" in df.columns:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=df["Age"], y=df["Heart Rate"], ax=ax)
    ax.set_xlabel("Age")
    ax.set_ylabel("Heart Rate")
    ax.set_title("Age vs. Heart Rate")
    st.pyplot(fig)
    st.markdown("📌 **Pros:** Helps identify trends and clusters in heart rate data.\n")
    st.markdown("📌 **Cons:** Can be hard to interpret if data is too scattered.")
else:
    st.error("⚠️ Required columns not found!")

st.subheader("⚖️ BMI Distribution (Histogram)")
if "BMI" in df.columns:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["BMI"], bins=20, kde=True, ax=ax)
    ax.set_xlabel("BMI")
    ax.set_ylabel("Frequency")
    ax.set_title("BMI Distribution Among Individuals")
    st.pyplot(fig)
    st.markdown("📌 **Pros:** Shows BMI frequency distribution.\n")
    st.markdown("📌 **Cons:** Doesn't indicate health conditions related to BMI.")
else:
    st.error("⚠️ 'BMI' column not found!")

st.subheader("🏃‍♂️ Exercise Intensity vs. Duration (Line Graph)")
if "Exercise Intensity" in df.columns and "Duration" in df.columns:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(x=df["Exercise Intensity"], y=df["Duration"], marker="o", ax=ax)
    ax.set_xlabel("Exercise Intensity")
    ax.set_ylabel("Duration (minutes)")
    ax.set_title("Exercise Intensity vs. Duration")
    st.pyplot(fig)
    st.markdown("📌 **Pros:** Shows trends between workout intensity and duration.\n")
    st.markdown("📌 **Cons:** Doesn't indicate effectiveness of exercises.")
else:
    st.error("⚠️ Required columns not found!")

st.subheader("🌦️ Weather Conditions vs. Calories Burned (Pie Chart)")
if "Weather Conditions" in df.columns and "Calories Burn" in df.columns:
    weather_groups = df.groupby("Weather Conditions")["Calories Burn"].sum()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(weather_groups, labels=weather_groups.index, autopct="%1.1f%%", startangle=140)
    ax.set_title("Weather Conditions Impact on Calories Burned")
    st.pyplot(fig)
    st.markdown("📌 **Pros:** Shows how weather impacts workout effectiveness.\n")
    st.markdown("📌 **Cons:** Doesn't account for exercise type.")
else:
    st.error("⚠️ Required columns not found!")

# st.success("✅ Data Visualizations Generated Successfully!")


#section 7: calories and diet
# Load food calorie dataset with caching
st.subheader("🍽️ Calorie Tracker & Diet Planning")

@st.cache_data
def load_food_data():
    df = pd.read_csv("calories_food.csv")

    # Ensure the "Calories" column is cleaned and numeric
    df["Calories"] = df["Calories"].astype(str).str.replace(r"\D+", "", regex=True).astype(float)
    
    return df

food_df = load_food_data()

# Function to calculate daily calorie needs
def calculate_calories(gender, age, weight, height, activity_level):
    if gender == "Male":
        bmr = 88.36 + (13.4 * weight) + (4.8 * height) - (5.7 * age)
    else:
        bmr = 447.6 + (9.2 * weight) + (3.1 * height) - (4.3 * age)
    return bmr * activity_level

# User inputs
gender = st.selectbox("Select Gender", ["Male", "Female"])
age = st.number_input("Enter Age", min_value=10, max_value=100, value=25)
weight = st.number_input("Enter Weight (kg)", min_value=30, max_value=200, value=70)
height = st.number_input("Enter Height (cm)", min_value=100, max_value=250, value=170)
activity_level = st.selectbox("Activity Level", ["Sedentary (1.2)", "Lightly Active (1.375)", "Moderately Active (1.55)", "Very Active (1.725)", "Super Active (1.9)"])
activity_map = {"Sedentary (1.2)": 1.2, "Lightly Active (1.375)": 1.375, "Moderately Active (1.55)": 1.55, "Very Active (1.725)": 1.725, "Super Active (1.9)": 1.9}

# Calculate daily calorie needs
daily_calories = calculate_calories(gender, age, weight, height, activity_map[activity_level])
st.write(f"### Your Estimated Daily Calorie Needs: {daily_calories:.2f} kcal")

# Initialize diet list in session state
if "diet" not in st.session_state:
    st.session_state.diet = []

# Food selection
selected_food = st.selectbox("Search Food", food_df["Food"].unique())
quantity = st.number_input("Enter Quantity (grams)", min_value=1, max_value=1000, value=100)

# Add food to diet
if st.button("Add to Diet"):
    matching_rows = food_df.loc[food_df["Food"] == selected_food, "Calories"]
    
    if not matching_rows.empty:
        calories = matching_rows.iloc[0] * (quantity / 100)  # Safe since Calories is float
        st.session_state.diet.append({"Food": selected_food, "Quantity": quantity, "Calories": calories})

# Display diet plan
st.subheader("🥗 Your Diet Plan")
if st.session_state.diet:
    diet_df = pd.DataFrame(st.session_state.diet)
    st.dataframe(diet_df)
    total_calories = diet_df["Calories"].sum()
    st.write(f"### Total Calories in Diet: {total_calories:.2f} kcal")

    # Remove food option
    food_to_remove = st.selectbox("Select Food to Remove", [item["Food"] for item in st.session_state.diet])

    if st.button("Remove Selected Food"):
        st.session_state.diet = [item for item in st.session_state.diet if item["Food"] != food_to_remove]
        st.success(f"Removed {food_to_remove} from your diet plan!")

else:
    st.write("No items added to diet.")
