from typing import List, Set

from pydantic import BaseModel


class OCRModel(BaseModel):
    coordinate: List
    result: Set


class Base64PostModel(BaseModel):
    base64_str: str