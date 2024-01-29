import os
import json
import psycopg2
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
from openai import OpenAI
import requests
import threading
from dotenv import find_dotenv, load_dotenv
from flask import Flask, request, jsonify, Response
from functions import keep_db_connection_stayin_alive, get_user, get_current_active_flights_from_chat, get_airports_from_chat, get_routes_from_chat
from flask_cors import CORS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory


load_dotenv(find_dotenv())
client = OpenAI()
flask_app = Flask(__name__)
auth = HTTPBasicAuth()
CORS(flask_app)

AVIATIONSTACK_API_KEY = os.environ["AVIATIONSTACK_API_KEY"]
aviationstack_base_url = "http://api.aviationstack.com/v1"
AIRLABS_API_KEY = os.environ["AIRLABS_API_KEY"]
airlabs_base_url = "https://airlabs.co/api/v9"
DATABASE_URL = os.environ["NEON_URL"]
connection = psycopg2.connect(DATABASE_URL)

stayin_alive_thread = threading.Thread(target=keep_db_connection_stayin_alive, args=(300,), daemon=True)
stayin_alive_thread.start()

@auth.verify_password
def verify_password(username, password):
    user = get_user(username)
    if user:
        user_id, user_name, password_hash = user
        if check_password_hash(password_hash, password):
            return user

@flask_app.route("/auth", methods=["POST"])    
def authenticate():
    credentials = request.json
    username = credentials.get('username')
    password = credentials.get('password')

    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400

    if verify_password(username, password):
        return jsonify({"message": "Authentication successful"}), 200
    else:
        return jsonify({"error": "Invalid credentials"}), 401

#test function
@flask_app.route("/datatest", methods=["GET"])
def get_data():
    """
    Fetch data from the database.
    Returns:
        list: A list of data or error message with status code.
    """
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM airports LIMIT 10")
    data = cursor.fetchall()
    cursor.close()
    return jsonify(data)

llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=0)

memory = ConversationBufferWindowMemory(k=10)
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False,
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_active_flights_from_chat",
            "description": "useful for when you need to fetch currently active flights based on optional filters input as keyword arguments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bbox": {
                        "type": "string",
                        "description": "Bounding box coordinates in the format \"SW Lat,SW Long,NE Lat,NE Long\".",
                    },
                    "zoom": {
                        "type": "integer",
                        "description": "Map zoom level to reduce the number of flights for rendering (0-11).",
                    },
                    "airline_iata": {
                        "type": "string",
                        "description": "Filtering by Airline IATA code.",
                    },
                    "flag": {
                        "type": "string",
                        "description": "Filtering by Airline Country ISO 2 code from Countries DB.",
                    },
                    "flight_number": {
                        "type": "string",
                        "description": "Filtering by Flight number only.",
                    },
                    "dep_iata": {
                        "type": "string",
                        "description": "Filtering by departure Airport IATA code.",
                    },
                    "arr_iata": {
                        "type": "string",
                        "description": "Filtering by arrival Airport IATA code.",
                    },
                },
            },
            "required": [],
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_airports_from_chat",
            "description": "useful for when you need to fetch airports based on optional filters input as keyword arguments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bboxes": {
                        "type": "string",
                        "description": "List of bounding box coordinates in the format 'xmin1,ymin1,xmax1,ymax1|xmin2,ymin2,xmax2,ymax2'.",
                    },
                    "airport_iatas": {
                        "type": "string",
                        "description": "Filtering by Airport IATA codes. Codes should be listed in comma separated string with no spaces. Example: 'LAX,JFK'."
                    },
                    "country_names": {
                        "type": "string",
                        "description": "Filtering by full country names. Example: 'United States,Canada'.",
                    },
                    "min_capacity": {
                        "type": "integer",
                        "description": "Filtering by minimum airport capacity.",
                    },
                    "max_capacity": {
                        "type": "integer",
                        "description": "Filtering by maximum airport capacity.",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Return top n airports by capacity.",
                    },
                    "bottom_n": {
                        "type": "integer",
                        "description": "Return bottom n airports by capacity.",
                    },
                    "airport_names": {
                        "type": "string",
                        "description": "Filtering by full airport names.",
                    },
                },
            },
            "required": [],
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_routes_from_chat",
            "description": "useful for when you need to fetch airline routes based on optional filters input as keyword arguments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "airline_iatas": {
                        "type": "string",
                        "description": "Filtering by Airline IATA codes. Codes should be listed in comma separated string. Example: 'AA,DL'."
                    },
                    "source_and_destination_airport_iatas": {
                        "type": "array",
                        "description": "Filtering by source and destination airport iata codes. Array of pairs of source and destination airport iatas. Example: [{source: 'LAX', destination: 'JFK'}, {source: 'AKL', destination: 'ZQN'}].",
                        "items": {
                            "type": "object",
                            "description": "keys are 'source' and 'destination', values are the airport iatas. Example: [{source: 'LAX', destination: 'JFK'}, {source: 'AKL', destination: 'ZQN'}]."
                        }
                    },
                    "source_airport_iatas": {
                        "type": "string",
                        "description": "An ordered list of all source airports from the query. Codes should be listed in comma separated string. Example: 'LAX,AKL'."
                    },
                    "destination_airport_iatas": {
                        "type": "string",
                        "description": "An ordered list of all destination airport iata codes. Codes should be listed in comma separated string. Example: 'JFK,ZQN'."
                    },
                    "source_and_destination_bboxes": {
                        "type": "array",
                        "description": "Array of pairs of source and destination bounding box coordinates in the format [{source: 'src1xmin,src1ymin,src1xmax,src1ymax', destination: 'dest1xmin,dest1ymin,dest1xmax,dest1ymax'}, {source: 'src2xmin,src2ymin,src2xmax,src2ymax', destination: 'dest2xmin,dest2ymin,dest2xmax,dest2ymax'}].",
                        "items": {
                            "type": "object",
                            "description": "keys are 'source' and 'destination', values are the bounding box coordinates. Example: [{source: 'src1xmin,src1ymin,src1xmax,src1ymax', destination: 'dest1xmin,dest1ymin,dest1xmax,dest1ymax'}, {source: 'src2xmin,src2ymin,src2xmax,src2ymax', destination: 'dest2xmin,dest2ymin,dest2xmax,dest2ymax'}]."
                        }
                    },
                    "source_bboxes": {
                        "type": "string",
                        "description": "List of source bounding box coordinates in the format 'xmin1,ymin1,xmax1,ymax1|xmin2,ymin2,xmax2,ymax2'."
                    },
                    "destination_bboxes": {
                        "type": "string",
                        "description": "List of destination bounding box coordinates in the format 'xmin1,ymin1,xmax1,ymax1|xmin2,ymin2,xmax2,ymax2'."
                    },
                    "source_and_destination_country_names": {
                        "type": "array",
                        "description": "Filtering by source and destination country names. List of pairs of source and destination country names. Example: [{source: 'United States', destination: 'Canada'}, {source: 'Japan', destination: 'China'}]'.",
                        "items": {
                            "type": "object",
                            "description": "keys are 'source' and 'destination', values are the country names. Example: [{source: 'United States', destination: 'Canada'}, {source: 'Japan', destination: 'China'}]'."
                        }
                    },
                    "source_country_names": {
                        "type": "string",
                        "description": "Filtering by only source country names. Country names should be listed in comma separated string. Example: 'United States,Canada'."
                    },
                    "destination_country_names": {
                        "type": "string",
                        "description": "Filtering by only destination country names. Country names should be listed in comma separated string. Example: 'United States,Canada'."
                    },
                    "number_of_stops": {
                        "type": "integer",
                        "description": "Use to fetch all routes and intermediate routes between 1 or more origins and destinations that include less than or equal to the specified number of stops."
                    }
                }
            },
            "required": [],
        }
    }
]

@flask_app.route("/geopt/flights", methods=["POST"])
@auth.login_required
def chat_geopt_flights():
    print(memory.load_memory_variables({}))
    """
    Receive user input and return geopt agent messages and actions.
    Returns:
        Response: A Flask response.
    """
    data = request.get_json()
    user_prompt = data.get("prompt")

    gpt_response = client.chat.completions.create(
        model="gpt-3.5-turbo-0613",
        messages=[{"role": "user", "content": user_prompt}],
        tools=tools,
        tool_choice="auto",
    )
    
    flask_app.logger.info(gpt_response)

    tool_calls = gpt_response.choices[0].message.tool_calls
    tool_response = None
    chat_response = None
    if tool_calls:
        arguments_dict = json.loads(gpt_response.choices[0].message.tool_calls[0].function.arguments)
        flask_app.logger.info(arguments_dict)
        function_name = gpt_response.choices[0].message.tool_calls[0].function.name
        if function_name == "get_current_active_flights_from_chat":
            tool_response = get_current_active_flights_from_chat(**arguments_dict)
            # return tool_response
        elif function_name == "get_airports_from_chat":
            tool_response = get_airports_from_chat(**arguments_dict)
            # return tool_response
        elif function_name == "get_routes_from_chat":
            tool_response = get_routes_from_chat(**arguments_dict)
            # return tool_response
    else:
        chat_response = str(gpt_response.choices[0].message.content)
        # return {"error": "No function found"}, 400

    print("TOOL_RESPONSE", tool_response)

    if isinstance(tool_response, Response):
        # Extract JSON data from the Response object
        tool_response_data = tool_response.get_json()
    else:
        # If it's already a JSON string
        tool_response_data = json.loads(tool_response) if isinstance(tool_response, str) else tool_response
    
    if (tool_response and len(str(tool_response_data)) < 1000):    
        memory.save_context({"input": user_prompt }, {"output": str(tool_response_data)})
    elif (tool_response and len(str(tool_response_data)) > 1000):
        truncated_result = str(tool_response_data)[:1000]
        memory.save_context({"input": user_prompt }, {"output": f"truncated result: {truncated_result}..."})
    
    if (chat_response == None and len(tool_response_data) > 0):
        followup_prompt = "Thanks! The results have been displayed on the map. Now please answer my question directly and conversationally and give a brief summary explaining the result."
    
        chat_response = conversation.predict(input=followup_prompt)
        memory.save_context({"input": followup_prompt}, {"output": chat_response})
    elif (chat_response == None and len(tool_response_data) == 0):
        followup_prompt = "Thanks! Now please answer my question directly and conversationally, and concisely state that the search yielded no results."
    
        chat_response = conversation.predict(input=followup_prompt)
        memory.save_context({"input": followup_prompt}, {"output": chat_response})
    else:
        memory.save_context({"input": user_prompt}, {"output": chat_response})

    print("CHAT RESPONSE: ", chat_response)

    return jsonify({
        "chat_response": chat_response,
        "tool_response": tool_response_data,
    })
