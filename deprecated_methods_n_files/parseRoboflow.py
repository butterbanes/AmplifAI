
import os

from dotenv import load_dotenv
from roboflow import Roboflow

"""
This method was originally used in the main AmplifAI program to parse a model from Roboflow
to detect chords visually. It seemed great on paper, but in practice it just significantly slowed
the fps of the video capture. I made sure it wasn't a computer issue by running it in the CSIS lab. I
believe that for some reason it is just very computationally expensive.

@author Ethan Smith
@version 11.10.2024
"""
def parse_rf_model():
    load_dotenv(dotenv_path=".env")
    api_key = os.getenv("API_KEY")
    project = Roboflow(api_key=api_key).workspace().project("chord-nxytx").version(1).model
    return project