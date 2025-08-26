import joblib
import pandas as pd

model = joblib.load('RandomForest.pkl')  
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')

X_train, _, _, _ = joblib.load('train_test_data.pkl') # Load training column order


categorical_columns = ['Gender', 'Trigger', 'Sweating', 'Shortness_of_Breath', 'Dizziness',
                       'Chest_Pain', 'Trembling', 'Medical_History', 'Medication', 'Smoking', 'Therapy']

numerical_columns = ['Age', 'Panic_Attack_Frequency', 'Duration_Minutes', 'Heart_Rate', 'Caffeine_Intake',
                     'Exercise_Frequency', 'Sleep_Hours', 'Alcohol_Consumption']

def get_user_input():
    print("\nPlease enter the following details:")
    
    age = int(input("Age: "))
    panic_attack_frequency = int(input("Panic Attack Frequency (per week): "))
    duration_minutes = float(input("Duration of Panic Attack (minutes): "))
    heart_rate = int(input("Heart Rate (bpm): "))
    caffeine_intake = int(input("Caffeine Intake (cups per day): "))
    exercise_frequency = int(input("Exercise Frequency (times per week): "))
    sleep_hours = float(input("Sleep Hours per night: "))
    alcohol_consumption = int(input("Alcohol Consumption (drinks per week): "))

    gender = input("Gender (Male/Female): ")
    trigger = input("Trigger (e.g., Stress, Caffeine, Unknown): ")
    sweating = input("Sweating (Yes/No): ")
    shortness_of_breath = input("Shortness of Breath (Yes/No): ")
    dizziness = input("Dizziness (Yes/No): ")
    chest_pain = input("Chest Pain (Yes/No): ")
    trembling = input("Trembling (Yes/No): ")
    medical_history = input("Medical History (e.g., Anxiety, None): ")
    medication = input("Currently on Medication? (Yes/No): ")
    smoking = input("Do you smoke? (Yes/No): ")
    therapy = input("Are you undergoing therapy? (Yes/No): ")

    return pd.DataFrame([{
        'Age': age,
        'Panic_Attack_Frequency': panic_attack_frequency,
        'Duration_Minutes': duration_minutes,
        'Heart_Rate': heart_rate,
        'Caffeine_Intake': caffeine_intake,
        'Exercise_Frequency': exercise_frequency,
        'Sleep_Hours': sleep_hours,
        'Alcohol_Consumption': alcohol_consumption,
        'Gender': gender,
        'Trigger': trigger,
        'Sweating': sweating,
        'Shortness_of_Breath': shortness_of_breath,
        'Dizziness': dizziness,
        'Chest_Pain': chest_pain,
        'Trembling': trembling,
        'Medical_History': medical_history,
        'Medication': medication,
        'Smoking': smoking,
        'Therapy': therapy
    }])

sample_data = get_user_input()

sample_encoded = encoder.transform(sample_data[categorical_columns])
sample_encoded_df = pd.DataFrame(sample_encoded, columns=encoder.get_feature_names_out(categorical_columns))

sample_data = sample_data.drop(columns=categorical_columns)
sample_data = pd.concat([sample_data, sample_encoded_df], axis=1)

sample_data[numerical_columns] = scaler.transform(sample_data[numerical_columns])

sample_data = sample_data[X_train.columns]

if 'ID' in sample_data.columns:
    sample_data = sample_data.drop(columns=['ID'])

prediction = model.predict(sample_data)

print(f"\n🔮 Predicted Panic Score: {prediction[0]:.2f}")
