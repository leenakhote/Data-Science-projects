from pathlib import Path
import os
import requests
import json
import pandas
import csv

def writeToCsvNew(filePath,csvData):
    with open(filePath+".csv", 'w') as csvfile:
        fieldnames = ['label', 'name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csvData)
        print("writing complete")

def collect_data():
    audio_data = []
    entries = Path('/home/tickled_media_7/leena/audio_classification/datasets/')
    for entry in entries.iterdir():
        for x in entry.iterdir():
            array_data = {}
            x = str(x)
            split_array = x.split('/')
            name = split_array[-1]
            label = split_array[-2]
            array_data["name"] = label+'/'+name
            array_data["label"] = label
            audio_data.append(array_data)
            # print(x)
    writeToCsvNew("/home/tickled_media_7/leena/audio_classification/classification/baby_audio", audio_data)

collect_data()