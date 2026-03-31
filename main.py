import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from datetime import datetime

# LOAD DATA

data = pd.read_csv("accident_data.csv")

# PREPARE DATA

X = data[['speed', 'weather', 'time', 'alcohol', 'traffic', 'road', 'experience']]
y = data['risk']

# TRAIN MODEL

model=LogisticRegression()
model.fit(X, y)

print("Model Trained Successfully!")

# MENU SYSTEM

while True:
    print("\n====== Accident Risk Prediction System ======")
    print("1. Predict Risk")
    print("2. Show History")
    print("3. Show Graph")
    print("4. Exit")
    print("5. See Previous Report")

    choice = input("Enter your choice: ")

    # OPTION 1: PREDICT
    
    if choice == '1':

        print("\nEnter details:")

        speed = int(input("Speed: "))
        weather = int(input("Weather (0=Clear, 1=Rainy, 2=Foggy): "))
        time = int(input("Time (0=Day, 1=Night): "))
        alcohol = int(input("Alcohol (0=No, 1=Yes): "))
        traffic = int(input("Traffic (0=Low, 1=Medium, 2=High): "))
        road = int(input("Road (0=City, 1=Highway, 2=Rural): "))
        experience = int(input("Driver Experience (0=Beginner, 1=Intermediate, 2=Experienced): "))

        # Validation
        if speed < 0 or speed > 200:
            print("Invalid speed!")
            continue

        # Prepare input
        input_data = pd.DataFrame([[speed, weather, time, alcohol, traffic, road, experience]],
                                 columns=['speed', 'weather', 'time', 'alcohol', 'traffic', 'road', 'experience'])

        # Prediction
        prob = model.predict_proba(input_data)
        risk_percent = prob[0][1] * 100

        print(f"\nAccident Risk: {risk_percent:.2f}%")

        # Risk category
        if risk_percent <= 20:
            risk_label = "🟢 Very Low Risk"
        elif risk_percent <= 40:
            risk_label = "🟢 Low Risk"
        elif risk_percent <= 60:
            risk_label = "🟡 Medium Risk"
        elif risk_percent <= 80:
            risk_label = "🟠 High Risk"
        else:
            risk_label = "🔴 Very High Risk"

        print(risk_label)

        # LABEL CONVERSION
        
        if weather == 0:
            weather_text = "Clear"
        elif weather == 1:
            weather_text = "Rainy"
        else:
            weather_text = "Foggy"

        time_text="Day" if time == 0 else "Night"
        alcohol_text = "No" if alcohol == 0 else "Yes"

        traffic_text = ["Low", "Medium", "High"][traffic]
        road_text = ["City", "Highway", "Rural"][road]
        exp_text = ["Beginner", "Intermediate", "Experienced"][experience]

        # SAFETY SUGGESTIONS
        
        print("\n--- Safety Suggestions ---")

        if speed > 80:
            print("⚠️ Reduce speed")
        if weather == 1:
            print("🌧️ Drive carefully in rain")
        if weather == 2:
            print("🌫️ Low visibility due to fog")
        if time == 1:
            print("🌙 Be cautious at night")
        if alcohol == 1:
            print("🚫 Do not drink and drive")
        if traffic == 2:
            print("🚗 Heavy traffic, stay alert")
        if road == 1:
            print("🛣️ Control speed on highway")
        if experience == 0:
            print("🧑‍🏫 Drive carefully (Beginner)")

        # SAVE HISTORY
        
        with open("history.txt", "a",encoding="utf-8") as f:
            f.write(f"Speed: {speed}, Weather: {weather_text}, Time: {time_text}, Alcohol: {alcohol_text}, Traffic: {traffic_text}, Road: {road_text}, Experience: {exp_text}, Risk: {risk_percent:.2f}% ({risk_label})\n")

        # GENERATE REPORT
        
        now = datetime.now()

        with open("report.txt", "w",encoding="utf-8") as f:
            f.write(f"""
                    
====== Accident Risk Report ======

Date & Time: {now}

Speed: {speed}
Weather: {weather_text}
Time: {time_text}
Alcohol: {alcohol_text}
Traffic: {traffic_text}
Road: {road_text}
Driver Experience: {exp_text}
---------------------------------
Risk Percentage: {risk_percent:.2f}%
Risk Level: {risk_label}

---------------------------------
Safety Suggestions:
""")

            if speed > 80:
                f.write("Reduce speed\n")
            if weather == 1:
                f.write("Drive carefully in rain\n")
            if weather == 2:
                f.write("Low visibility due to fog\n")
            if time == 1:
                f.write("Be cautious at night\n")
            if alcohol == 1:
                f.write("Do not drink and drive\n")
            if traffic == 2:
                f.write("Heavy traffic, stay alert\n")
            if road == 1:
                f.write("Control speed on highway\n")
            if experience == 0:
                f.write("Drive carefully (Beginner)\n")

        print("\n📄 Report generated successfully (report.txt)")

    # OPTION 2: HISTORY
    
    elif choice == '2':
        print("\n--- History ---")
        try:
            with open("history.txt", "r", encoding="utf-8") as f:
                print(f.read())
        except:
            print("No history found.")

    # OPTION 3: GRAPH
    
    elif choice == '3':
        probabilities = model.predict_proba(X)
        risk_values = probabilities[:, 1] * 100

        plt.scatter(data['speed'], risk_values)
        plt.xlabel("Speed")
        plt.ylabel("Risk (%)")
        plt.title("Speed vs Accident Risk (%)")
        plt.show()

    # OPTION 5: SHOW REPORT
    
    elif choice=='5':
        print("\n--- Previous Report ---")
        try:
            with open("report.txt", "r", encoding="utf-8") as f:
                print(f.read())
        except:
            print("No report found.")

    # OPTION 4: EXIT
    
    elif choice == '4':
        print("Exiting program...")
        break

    else:
        print("Invalid choice. Try again.")