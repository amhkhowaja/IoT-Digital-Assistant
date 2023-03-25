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
#Actions

class ActionCPIlink(Action):

    def name(self) -> Text:
        return "action_CPI_link"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        prediction = tracker.latest_message


        # client = pymongo.MongoClient("mongodb://admin:password@mongo-db:27017/")
        client = pymongo.MongoClient("mongodb://localhost:27017")
        db = client["IOTA"]
        CPI_links = db["CPI"]

        try:
            intent_1 = prediction['intent'].get('name')
            current_entity_1 = prediction['entities'][0]['entity']
        except IndexError:
            current_entity_1 = None

        if current_entity_1 is None:
            msg=f"Sorry I couldn't get it. Could you please rephrase?"
            dispatcher.utter_message(text=msg)
            return []

        entity_value = next(tracker.get_latest_entity_values(current_entity_1), None)
        document = CPI_links.find({'$and':[{'intent':intent_1},{'sub_entities':entity_value}]})
        link = document[0].get('enterprise_links')

        if link is None:
#            dispatcher.utter_message(text=current_entity)
            msg=f"The detected Entity is {current_entity_1} but unfortunately no CPI link exists for {entity_value_1}. Please rephrase the query."
            dispatcher.utter_message(text=msg)
            return []

        msg=f"Please follow the below link for detailed information:"
        dispatcher.utter_message(text=msg)


#        data2 = {
#                    "payload":"pdf_attachment",
#                    "title": "Relevant documentation",
#                     "title": entity_value,
#                    "url": link
#                }
#
#
#        dispatcher.utter_message(json_message=data2)


        data1 = {
           "payload": 'iFrame',
           "data": [
                       {
                           "image": "https://b.zmtcdn.com/data/pictures/1/18602861/bd2825ec26c21ebdc945edb7df3b0d99.jpg",
                        #    "title": "Relevant documentation",
                           "title": entity_value,
                           "ratings": "4.5",
                           "url":link
                       }
               ]
           }

        dispatcher.utter_message(json_message=data1)
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
            client = MongoClient("mongodb://localhost:27017")
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
        
# class ActionIMSIStats(Action):

#     def name(self) -> Text:
#         return "action_IMSI_Stats"

#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

#         prediction = tracker.latest_message
# #        dispatcher.utter_message(text="Entered Custom Actions Abrar")
#         slot_value = tracker.get_slot("IMSI_number")

# # Fetching Entity Values
#         default_entity_value="IMSI_number"
#         try:
# #           intent_1 = prediction['intent'].get('name')
#             current_entity_1 = prediction['entities'][0]['entity']
#             entity_value = next(tracker.get_latest_entity_values(current_entity_1), None)
#         except IndexError:
#             current_entity_1 = None

#         # if (current_entity_1 is None) or (current_entity_1==default_entity_value) or (entity_value=='imsi'):
#         if (current_entity_1=="subscription") or (current_entity_1 is None) or (current_entity_1==default_entity_value) or (entity_value=='imsi'):
#                 sql="select "+ "INSTALLATION_DATE, SIM_SUBSCRIPTION_STATE, MSISDN, PIN1, PUK1, SIM_STATUS" +" from SUBSCRIPTION where IMSI="+slot_value
#         else:
#                 sql="select "+ entity_value +" from SUBSCRIPTION where IMSI="+slot_value

# #        dispatcher.utter_message(text=sql)

#         connection = None
#         query_result = ""

#         try:
#             with cx_Oracle.connect(user="smapp", password="3928ad5894cf58dc7569ad5b733c85",
#                                        dsn="acc50-dbh-scan.dcp.fi.eu.xdn.ericsson.se:1521/APCONT_SM_SERVICE",
#                                        encoding="UTF-8") as connection:
#                 with connection.cursor() as cursor:
#                     cursor.execute(sql)
#                     while True:
#                         row = cursor.fetchone()
#                         if row is None:
#                             break
#                         query_result=row
# #                        dispatcher.utter_message(row[0])

#         except cx_Oracle.Error as error:
#             dispatcher.utter_message(text=error)
#         if query_result is None:
#             dispatcher.utter_message(text="Sorry no record found for the given IMSI number.")
# #        finally:
# #            pass
# #            if connection:
# #                connection.close()
#         # if (current_entity_1 is None) or (current_entity_1==default_entity_value) or (entity_value=='imsi'):
#         if (current_entity_1=="subscription") or (current_entity_1 is None) or (current_entity_1==default_entity_value) or (entity_value=='imsi'):
#             msg=f"""The details of IMSI {slot_value} are:
#             The Installation date is {query_result[0]}.
#             The SIM state is {query_result[1]}.
#             MSISDN is  {query_result[2]}.
#             Pin is {query_result[3]}.
#             Puk is  {query_result[4]}.
#             And SIM status is {query_result[5]}."""
#         else:
#             msg=f"The {' '.join(str(entity_value).lower().split('_'))} of IMSI {slot_value} is: \n {query_result[0]}"
#         dispatcher.utter_message(text=msg)

#         return[]


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
        dispatcher.uttter_message(text="Perfect!")
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
            dispatcher.uttter_message(text="Perfect!")
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
            client = MongoClient("mongodb://localhost:27017")
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
            client = MongoClient("mongodb://localhost:27017")
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
        attributes = ["msisdn","plan_name", "connectivity_lock", "network_connectivity", "in_session", "billing_state", "monthly_data", "data_trend"]
        try:
            client = MongoClient("mongodb://localhost:27017")
            db = client["IOTA"]
            inventory = db["inventory"]
        except:
            dispatcher.utter_message(text="Sorry! we can not build the connection with the database.")
            return []
        # extracted entities names:
        extracted_entities =  prediction["entities"]
        extracted_entities = [entity for entity in extracted_entities if entity in attributes]
        all_entities = [entity["entity"] for entity in extracted_entities ]
        #if msisdn number is not given then it can not perform correctly
        if "msisdn" not in all_entities:
            dispatcher.utter_message(template="utter_ask_msisdn") 
            return [SlotSet("requested_slot", "msisdn"), FollowupAction("action_listen")]
        msisdn_slot = tracker.get_slot("msisdn")
        try:
            result = list(inventory.find({"msisdn": msisdn_slot}))[0]
        except IndexError as e:
            dispatcher.utter_message(text="Sorry! we cannot find that msisdn \{msisdn\} in our data.")
            dispatcher.utter_message(template="utter_ask_msisdn")
            return [SlotSet("requested_slot", "msisdn"), FollowupAction("action_listen")]
        # nlu: change the [connectivity lock]{"entity": "inventory_attribute", "value": "connectivity_lock", role:"update_attribute"} of [msisdn]{"entity": "inventory_attribute", "value": "msisdn", "role": "fetch_attribute"} [23123123123]{"entity": "msisdn", "value": "23123123123", "role": "fetch_value"} to [locked]{"entity": "connectivity_lock", "value": "locked", "role":"update_value"}
        # using pandas dataframe for manipulation in the data and to query the data
        df= pd.read_json(str(json.dumps(extracted_entities)), orient="records")
        

        #queries:
        update = {}
        fetch = {}
        for i, row in df[df["role"]=="update_value"].reset_index().iterrows():
            update[row["entity"]] = df[df["role"]=="update_value"].iloc[i]["value"] 
        for i, row in df[df["role"]=="fetch_value"].reset_index().iterrows():
            fetch[row["entity"]] = df[df["role"]=="fetch_value"].iloc[i]["value"] 
        
        print("update: "+update, " fetch: "+fetch)

        if "msisdn" in update:
            dispatcher.utter_message(text="MSISDN cannot be updated. ")
            return []
        if "msisdn" not in fetch:
            dispatcher.utter_message(text="MSISDN is required to update the database: ")
            return []
        

        #updating the database
        result = inventory.update_one(fetch, {"$set":update})
        if result.modified_count > 0:
            dispatcher.utter_message(text="Hurrah. Updated Successfully.")
        else : 
            dispatcher.utter_message(text="Unfortunately we cannot update your data" )

        return []
        #      
        # extracted entities and their values as
        # we will need to define the roles for updation
        # we cannot update the msisdn number
        # given msisdn number we can update the previous value to a new value of any attribute
        # user : change the connectivity lock of msisdn 23123123123 to locked 
        # user : update the monthly package from 10 gb europe plan to 20 gb america plan
        # user: activate the 55158481515 in the inventory
        # considering the example as above and format of nlu :
            # change the [connectivity lock]{"entity": "inventory_attribute", "value": "connectivity_lock", role:"update_attribute"} of [msisdn]{"entity": "inventory_attribute", "value": "msisdn", "role": "fetch_attribute"} [23123123123]{"entity": "msisdn", "value": "23123123123", "role": "fetch_value"} to [locked]{"entity": "connectivity_lock", "value": "locked", "role":"update_value"}
        