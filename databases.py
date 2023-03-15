import pymongo

def create_database(name):
    query = "create database " + name  