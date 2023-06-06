from typing import Any, Text, Dict, List, Union
from rasa_sdk import Action, Tracker
# from rasa.core.actions.form_validation import FormValidationAction
from rasa_sdk import Tracker
from rasa_sdk.forms import FormValidationAction
from rasa_sdk.events import SlotSet, FollowupAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
import pymongo
from pymongo import MongoClient
from pymongo.errors import WriteError
import cx_Oracle
import requests
import json
import random
import re
import pandas as pd
import json
import people_also_ask as paa
#Actions

class ActionCPIlink(Action):

    def name(self) -> Text:
        return "action_CPI_link"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        prediction = tracker.latest_message
        query = prediction["text"]
        if 'IOT' not in query.upper():
            query = query + "in IoT"
        
        response = paa.get_answer(query)
        msg=""
        buttons=[]
        if response["has_answer"] == False and len(response["related_questions"])>0:
            msg="Could you please rephrase it? Below are the related questions."
            buttons = [{"title":button, "payload":"/"+button} for button in response["related_questions"]]
        elif response["has_answer"] == False and len(response["related_questions"])==0:
            msg="I am really sorry, I cannot answer this kind of question. Please ask relavant questions."
        else :
            msg=response["response"]
        dispatcher.utter_message(text=msg, buttons=buttons)
        


#        data2 = {
#                    "payload":"pdf_attachment",
#                    "title": "Relevant documentation",
#                     "title": entity_value,
#                    "url": link
#                }
#
#
#        dispatcher.utter_message(json_message=data2)


        # data1 = {
        #    "payload": 'iFrame',
        #    "data": [
        #                {
        #                    "image": "https://b.zmtcdn.com/data/pictures/1/18602861/bd2825ec26c21ebdc945edb7df3b0d99.jpg",
        #                 #    "title": "Relevant documentation",
        #                    "title": entity_value,
        #                    "ratings": "4.5",
        #                    "url":link
        #                }
        #        ]
        #    }

        # dispatcher.utter_message(json_message=data1)
        #aadarsh changes

        return []

class ActionIMSIStats(Action):
    def name(self) -> Text:
        return "action_IMSI_Stats"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        prediction = tracker.latest_message
        slot_value = tracker.get_slot("IMSI_number")
        default_entity_value="IMSI_number"
        try:
            client = MongoClient("mongodb://mongo:27017")
            db = client["IOTA"]
            subscription_details = db["subscription_details"]
        except:
            dispatcher.utter_message(text="Sorry! we can not build the connection with the database")
            return []
        try:
#           intent_1 = prediction['intent'].get('name')
            current_entity_1 = prediction['entities'][0]['entity']
            entity_value = next(tracker.get_latest_entity_values(current_entity_1), None)
        except IndexError:
            current_entity_1 = None
        query = {'imsi': int(slot_value)}
        # if (current_entity_1 is None) or (current_entity_1==default_entity_value) or (entity_value=='imsi'):
        try:
            if (current_entity_1=="subscription") or (current_entity_1 is None) or (current_entity_1==default_entity_value) or (entity_value=='imsi'):
                # details="select "+ "INSTALLATION_DATE, SIM_SUBSCRIPTION_STATE, MSISDN, PIN1, PUK1, SIM_STATUS" +" from SUBSCRIPTION where IMSI="+slot_value
                query_result = list(subscription_details.find(query))[0]
            else:
                # sql="select "+ entity_value +" from SUBSCRIPTION where IMSI="+slot_value
                query_result = list(subscription_details.find(query))[0][entity_value]
        except IndexError as error:
            dispatcher.utter_message(text = "No data found for the IMSI number \""+slot_value+"\"")
            return []
        if (current_entity_1=="subscription") or (current_entity_1 is None) or (current_entity_1==default_entity_value) or (entity_value=='imsi'):
            msg=f"""The details of IMSI {slot_value} are:
            The Installation date is {query_result["Installation_date"]}.
            The SIM subscription state is {query_result["sim_subscription_state"]}.
            MSISDN is  {query_result["msisdn"]}.
            Pin is {query_result["pin1"]}.
            Puk is  {query_result["puk1"]}.
            And SIM status is {query_result["sim_status"]}."""
        else:
            msg=f"The {' '.join(str(entity_value).lower().split('_'))} of IMSI {slot_value} is: \n {query_result[entity_value]}"
        dispatcher.utter_message(text=msg)
        return []
        


class ActionNewsFetch(Action):
    def name(self) -> Text:
        return "action_news_fetch"
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) ->List[Dict[Text, Any]]:
        prediction = tracker.latest_message
        query_params = {
            "source": "bbc-news",
            "sortBy": "top",
            "apiKey": "4dbc17e007ab436fb66416009dfb59a8"
            }
        main_url = " https://newsapi.org/v1/articles"
        # fetching data in json format
        try:
            req = requests.get(main_url, params=query_params)
            open_bbc_page = req.json()
            news=[]
            dispatcher.utter_message(text="Here is some headlines from {}".format(open_bbc_page["source"]))
            for i, val in enumerate(open_bbc_page["articles"]):
                # headline="{}. {}\nRead More about it here : {}".format(i, val["title"], val["url"])
                headline = {
                    "payload": 'cardsCarousel',
                    "data": [
                                {
                                    "image": val["urlToImage"],
                                    #    "title": "Relevant documentation",
                                    "title": val["title"],
                                    "ratings": "4.5",
                                    "url":val["url"]
                                }
                        ]
                    }
                news.append(headline)
            rand = random.randint(0, len(open_bbc_page['articles']))
            dispatcher.utter_message(json_message=news[rand])
        except (requests.exceptions.RequestException, KeyError) as e:
            dispatcher.utter_message(text="Sorry! unable to connect BBC news. Please try again.")
        return []

class ValidateCustomerTypeForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_customer_type_form"
    
    @staticmethod
    def customer_type_db() -> List[Text]:
        return ["enterprise", "advanced_reseller"] # advanced reseller
    
    def validate_customer_type(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        if slot_value.lower() in self.customer_type_db():
            # validation succeeded, set the value of the "cuisine" slot to value
            return {"customer_type": slot_value}
        else:
            # validation failed, set this slot to None so that the
            # user will be asked for the slot again
            return {"customer_type": None}

class ValidateEnterpriseForm(FormValidationAction):
    def name(self):
        return "validate_enterprise_form"
    
    @staticmethod
    def parent_organization_db():
        return ["china_telecom_mongolia_branch"]
    
    def validate_enterprise_name(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        return {'enterprise_name': slot_value}

    def validate_enterprise_agreement_number(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        enterprise_agreement_number = slot_value.lower()
        if len(enterprise_agreement_number) != 14:
            dispatcher.utter_message(template="utter_invalid_enterprise_agreement_number")
            return {"enterprise_agreement_number": None}
        dispatcher.utter_message(text="Perfect!")
        return {"enterprise_agreement_number": slot_value}
    
    def validate_parent_organization(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        parent_organization = slot_value.lower()
        if parent_organization == "china_telecom_mongolia_branch" and parent_organization.lower() in self.parent_organization_db():
            parent_organization = "ChinaTelecomMongoliabranch"
            dispatcher.utter_message(text="Perfect!")
        else:
            parent_organization = None
        return {'parent_organization': parent_organization}

class ActionFetchInventory(Action):
    def name(self) -> Text:
        return "action_fetch_inventory"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        prediction = tracker.latest_message
        # slot_value = tracker.get_slot("IMSI_number")
        attributes = ["msisdn","plan_name", "connectivity_lock", "network_connectivity", "in_session", "billing_state", "monthly_data", "data_trend"]

        try:
            client = MongoClient("mongodb://mongo:27017")
            db = client["IOTA"]
            inventory = db["inventory"]
        except:
            dispatcher.utter_message(text="Sorry! we can not build the connection with the database.")
            return []
        # Fetching all the data from the inventory database
        all_entities = [entity["entity"] for entity in prediction["entities"]] #list
        print(str(prediction))
        all_entities = [entity for entity in all_entities if entity in attributes]
        
        entities = {}
        for i in all_entities:
            print(str(i))
            entities[i] = next(tracker.get_latest_entity_values(i), None)
        filtered_entities = {k: int(v) if v.isnumeric() else {'$regex':'^'+v+'$', '$options':'i'} for k, v in entities.items() if v is not None}
        # I am assuming that in most of queries the none entity will be main attribute which will be asked
        none_entities = {k:v for k,v in entities.items() if v is None}

        print("Entities"+ str(entities))
        print("Filtered_entities: "+str(filtered_entities))
        print("None Entities"+ str(none_entities))
        try:
            query_result = list(inventory.find(filtered_entities))
        except IndexError as e:
            dispatcher.utter_message(text="Sorry! there is no data on that one")
            return []
        msg = f"""Sure, Here is the data for AND[{entities}]=
        {query_result}      
        """
        dispatcher.utter_message(text=msg)
        return []
    
class SubmitOnboardingForm(Action):
    def name(self) -> Text:
        return "action_submit_onboarding"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        prediction = tracker.latest_message

        
        #connecting to mongodb
        try:
            client = MongoClient("mongodb://mongo:27017")
            db = client["IOTA"]
            customers = db["customers"]
        except:
            dispatcher.utter_message(text="Sorry! we can not build the connection with the database.")
            return []
        
        required_entities = ["customer_type","enterprise_name", "enterprise_agreement_number", "parent_organization"]
        all_slots= {slot:value for slot,value in tracker.slots if slot in required_entities}
        try:
            result = customers.insert_one(all_slots)
        except WriteError as e:
            dispatcher.utter_message(text = "There is unexpected uploading error coming up. Could you please try onboarding after some time.")
            return []
class ActionUpdateInventory(Action):
    def name(self) -> Text:
        return "action_update_inventory"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        prediction = tracker.latest_message
        print(prediction)
        attributes = ["msisdn","plan_name", "connectivity_lock", "network_connectivity", "in_session", "billing_state", "monthly_data", "data_trend"]
        try:
            client = MongoClient("mongodb://mongo:27017")
            db = client["IOTA"]
            inventory = db["inventory"]
        except:
            dispatcher.utter_message(text="Sorry! we can not build the connection with the database.")
            return []
        # extracted entities names:
        extracted_entities =  prediction["entities"]
        # extracted_entities = [entity for entity in extracted_entities if entity in attributes]
        all_entities = [entity["entity"] for entity in extracted_entities if entity["entity"] in attributes]
        print("All Entities:", all_entities)
        #if msisdn number is not given then it can not perform correctly
        if "msisdn" not in all_entities:
            dispatcher.utter_message(template="utter_ask_msisdn") 
            return [SlotSet("requested_slot", "msisdn"), FollowupAction("action_listen")]
        # msisdn_slot = tracker.get_slot("msisdn")
        msisdn_slot = int([ent.get("value") for ent in extracted_entities if ent["entity"]=="msisdn"][0])
        try:
            result = list(inventory.find({"msisdn": msisdn_slot}))[0]
        except IndexError as e:
            dispatcher.utter_message(text="Sorry! we cannot find that msisdn \{msisdn\} in our data.")
            dispatcher.utter_message(template="utter_ask_msisdn")
            return [SlotSet("requested_slot", "msisdn"), FollowupAction("action_update_inventory")]
        # nlu: change the [connectivity lock]{"entity": "inventory_attribute", "value": "connectivity_lock", role:"update_attribute"} of [mLdn]{"entity": "inventory_attribute", "value": "msisdn", "role": "fetch_attribute"} [23123123123]{"entity": "msisdn", "value": "23123123123", "role": "fetch_value"} to [locked]{"entity": "connectivity_lock", "value": "locked", "role":"update_value"}
        # using pandas dataframe for manipulation in the data and to query the data
        df= pd.read_json(str(json.dumps(extracted_entities)), orient="records")
        

        #queries:
        update = {}
        fetch = {}
        for i, row in df[df["role"]=="update_value"].reset_index().iterrows():
            update[row["entity"]] = df[df["role"]=="update_value"].iloc[i]["value"] 
        for i, row in df[df["role"]=="fetch_value"].reset_index().iterrows():
            fetch[row["entity"]] = df[df["role"]=="fetch_value"].iloc[i]["value"] 
        
        print("update: "+str(update), " fetch: "+str(fetch))

        if "msisdn" in update:
            dispatcher.utter_message(text="MSISDN cannot be updated. ")
            return []
        if "msisdn" not in fetch:
            dispatcher.utter_message(text="MSISDN is required to update the database: ")
            return []
        else:
            fetch["msisdn"]=int(fetch["msisdn"])

        #updating the database
        result = inventory.update_one(fetch, {"$set":update})
        if result.modified_count > 0:
            dispatcher.utter_message(text="Hurrah. Updated Successfully.")
        else : 
            dispatcher.utter_message(text="It seems like you already have it updated" )

        return []
        #      
        # extracted entities and their values as
        # we will need to define the roles for updation
        # we cannot update the msisdn number
        # given msisdn number we can update the previous value to a new value of any attribute
        # user :  
        # user : update the monthly package from 10 gb europe plan to 20 gb america plan
        # user: activate the 55158481515 in the inventory
        # considering the example as above and format of nlu :
            # change the [connectivity lock]{"entity": "inventory_attribute", "value": "connectivity_lock", role:"update_attribute"} of [msisdn]{"entity": "inventory_attribute", "value": "msisdn", "role": "fetch_attribute"} [23123123123]{"entity": "msisdn", "value": "23123123123", "role": "fetch_value"} to [locked]{"entity": "connectivity_lock", "value": "locked", "role":"update_value"}
        # update the package [monthly limit]{"entity": "inventory_attribute", "value": "monthly_data"} from [10gb]{"entity": "monthly_data", "role": "fetch_value", "value": "10 GB"} to [20gb]{"entity": "plan_name", "role": "update_value", "value": "20 GB"} of [mobile number]{"entity": "inventory_attribute", "value": "msisdn"} [12345678901]{"entity": "msisdn", "role": "fetch_value"}