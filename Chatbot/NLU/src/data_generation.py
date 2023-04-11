#!/usr/bin/env python
'''
Data files generation in json , yaml and csv

'''
# importing the libraries
import pandas as pd
import json
import yaml


def import_dataframe(file_name: str):
    '''import the dataframe from file and preprocess the columns for data extraction'''
    df = pd.read_csv(file_name)
    df["Questions"] = df["Questions"].map(
        lambda x: "".join([a for a in x if not (a == '(' or a == ')')])
    )
    df["Sub-Entities"] = df["Sub-Entities"].map(
        lambda x: x.strip("]['").split("', '"))
    df["Sub-Entities"] = df["Sub-Entities"].map(
        lambda x: [a.strip() for a in x])
    df["Main_Entities"] = df["Main_Entities"].map(
        lambda x: x.strip("]['").split("', '")
    )
    df["Main_Entities"] = df["Main_Entities"].map(
        lambda x: [a.strip() for a in x])
    df = df[["Questions", "Intent", "Sub-Entities", "Main_Entities"]]
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def preprocess_dataframe(dataframe):
    '''import the dataframe from file and preprocess the columns for data extraction'''
    df = dataframe
    df["Questions"] = df["Questions"].map(
        lambda x: "".join([a for a in x if not (a == '(' or a == ')')])
    )
    df["Sub-Entities"] = df["Sub-Entities"].map(
        lambda x: x.strip("]['").split("', '"))
    df["Sub-Entities"] = df["Sub-Entities"].map(
        lambda x: [a.strip() for a in x])
    df["Main_Entities"] = df["Main_Entities"].map(
        lambda x: x.strip("]['").split("', '")
    )
    df["Main_Entities"] = df["Main_Entities"].map(
        lambda x: [a.strip() for a in x])
    df = df[["Questions", "Intent", "Sub-Entities", "Main_Entities"]]
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def generate_json_dict(intent, entities, question, intents={}):
    '''generate the json dictionary categorizing the intent and then
    their entity and their examples'''
    if intent in intents:
        for e in entities:
            if e in intents[intent]["entities"]:
                intents[intent]["entities"][e].append(question)
            else:
                intents[intent]["entities"][e] = [question]
    else:
        ent = {}
        for e in entities:
            if e not in ent:
                ent[e] = []
            ent[e].append(question)
        intents[intent] = {"entities": ent}
    return intents


def add_entities(question, sub, ent):
    '''Helper function for entities_into_questions() for reformating the questions
    for the yaml file'''
    count = 0
    for i, j in enumerate(sub):
        if j in question.lower():
            question = question.lower().replace(
                j, "[{sube}]({enti})".format(sube=j, enti=ent[i])
            )
            count = 1
    return question, count


def entities_into_questions(df):
    '''It will iterate all the questions and format quesions with entity name and its value like
        [VALUE](ENTITY)
    '''
    count = 0
    que = []
    for _, row in df.iterrows():
        questions = row["Questions"]
        sub = row["Sub-Entities"]
        ents = row["Main_Entities"]
        add = add_entities(questions, sub, ents)
        count += add[1]
        que.append(add[0])
    return que, count


def generate_json(df):
    '''Generate the json file for the data'''
    data = {}
    for _, entry in df.iterrows():
        data = generate_json_dict(
            entry["Intent"], entry["Sub-Entities"], entry["Questions"], data
        )
    with open("../data/data.json", "w") as file:
        json.dump(data, file, indent=4)
    # return data


def generate_intent_csv(df):
    '''Generate the csv file for only questions and intents'''
    fl = df[["Questions", "Intent"]]
    fl.to_csv("../data/ques_int.csv", index=False)


def generate_yaml(df):
    '''Generate the Yaml file for the intent classification'''
    yamls = {}
    df.insert(4, "Questions_with_entities", entities_into_questions(df)[0])
    for i, entry in df.iterrows():
        if entry["Intent"] not in yamls:
            yamls[entry["Intent"]] = [entry["Questions_with_entities"]]
        else:
            yamls[entry["Intent"]].append(entry["Questions_with_entities"])
    del df["Questions_with_entities"]
    nlu = {}
    nlus = []
    for intent in yamls:
        nlu["intent"] = intent
        nlu["examples"] = yamls[intent]
        a = nlu.copy()
        nlus.append(a)
    yamls = {"nlu": nlus}
    with open("../data/data.yaml", mode="w") as file:
        yaml.dump(yamls, file, indent=1, sort_keys=False,
                  default_flow_style=False)



if __name__ == "__main__":
    file_name = "../data/final_data.csv"
    df = import_dataframe(file_name)
    generate_json(df)
    generate_yaml(df)
    generate_intent_csv(df)
    print("Files Generated")
