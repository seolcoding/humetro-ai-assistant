from typing import Union
import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()


app.add_middleware(
        CORSMiddleware,
)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/langchain")
def call_langchain(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
