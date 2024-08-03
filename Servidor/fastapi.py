from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}

# Punto final para obtener recomendaciones
@app.get("/recommendations")
def get_recommendations(title: str = None, user_id: int = None, n_recommendations: int = 10):
    from Modelo.model import get_real_time_recommendations
    return get_real_time_recommendations(title=title, user_id=user_id, n_recommendations=n_recommendations)
