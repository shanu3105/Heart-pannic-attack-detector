from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

model = joblib.load("models/RandomForest.pkl")
encoder = joblib.load("models/encoder.pkl")
scaler = joblib.load("models/scaler.pkl")

templates = Jinja2Templates(directory="templates")

categorical_columns = ['Gender', 'Trigger', 'Sweating', 'Shortness_of_Breath', 'Dizziness',
                       'Chest_Pain', 'Trembling', 'Medical_History', 'Medication', 'Smoking', 'Therapy']

numerical_columns = ['Age', 'Panic_Attack_Frequency', 'Duration_Minutes', 'Heart_Rate', 'Caffeine_Intake',
                     'Exercise_Frequency', 'Sleep_Hours', 'Alcohol_Consumption']

@app.get("/", response_class=HTMLResponse)
async def serve_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict(request: Request):
    try:
        data = await request.json()  

        df = pd.DataFrame([data])

        encoded_features = encoder.transform(df[categorical_columns])
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

        df = df.drop(columns=categorical_columns)
        df = pd.concat([df, encoded_df], axis=1)

        df[numerical_columns] = scaler.transform(df[numerical_columns])

        prediction = model.predict(df)[0]

        return JSONResponse(content={"prediction": round(prediction, 2)})

    except Exception as e:
        return JSONResponse(content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
