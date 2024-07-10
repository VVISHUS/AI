from fastapi import FastAPI

app=FastAPI()


food_items={
    'indian': ["Samosa","Dosa"],
    'american': ["Hot Dog","Apple Pie"]
}
@app.get("/get_fooditems/{cuisine}")

async def get_items(cuisine):
    return food_items.get(cuisine)