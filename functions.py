import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import requests
from langchain.agents import tool
import psycopg2
from psycopg2 import pool
from flask import Flask, request, jsonify
import threading
import time

load_dotenv(find_dotenv())
AIRLABS_API_KEY = os.environ["AIRLABS_API_KEY"]
airlabs_base_url = "https://airlabs.co/api/v9"
DATABASE_URL = os.environ["NEON_URL"]
db_pool = pool.ThreadedConnectionPool(1, 20, DATABASE_URL)

def get_db_connection():
    return db_pool.getconn()

def return_db_connection(conn):
    db_pool.putconn(conn)
    
def keep_db_connection_stayin_alive(interval=300):
    """
    Periodically executes a trivial query to keep the database connection alive.
    :param interval: Interval in seconds between keep-alive queries.
    """
    while True:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            return_db_connection(conn)
        except Exception as e:
            print(f"Error keeping the DB connection alive: {e}")
        time.sleep(interval)
        
def get_user(username):
    """
    Fetch user from the database by username.
    """
    for _ in range(3):  # Retry up to 3 times
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            result = cursor.fetchone()
            print(result)
            cursor.close()
            return_db_connection(conn)
            return result
        except psycopg2.OperationalError as e:
            print(f"OperationalError occurred: {e}")
            time.sleep(1)  # Wait for 1 second before retrying
    return None

def get_current_active_flights_from_chat(**kwargs):
    """
    Fetch currently active flights based on various optional filters input as kwargs.
    
    Optional Parameters:
    - bbox (str): Bounding box coordinates in the format "SW Lat,SW Long,NE Lat,NE Long".
    - zoom (int): Map zoom level to reduce the number of flights for rendering (0-11).
    - airline_iata (str): Filtering by Airline IATA code.
    - flag (str): Filtering by Airline Country ISO 2 code from Countries DB.
    - flight_number (str): Filtering by Flight number only.
    - dep_iata (str): Filtering by departure Airport IATA code.
    - arr_iata (str): Filtering by arrival Airport IATA code.
    
    Returns:
        list: A list of active flights or error message with status code.
    """
    
    url = f"{airlabs_base_url}/flights"
    # Start with mandatory API key and then update with any optional parameters provided
    params = {
        "api_key": AIRLABS_API_KEY,
    }
    params.update(kwargs)  # Add any other provided optional parameters

    # Make the GET request to the API
    response = requests.get(url, params=params)

    # Check if the response was successful
    if response.status_code == 200:
        # Parse the flights from the response
        flights = response.json().get('response', [])
        return flights
    else:
        # Handle errors
        print("Failed to retrieve data:", response)
        return response.json(), response.status_code
    
    
def get_airports_from_chat(**kwargs):
    """
    Fetch airports based on various optional filters input as kwargs.
    
    Optional parameters:
    - bboxes (str): List of bounding box coordinates in the format 'xmin1,ymin1,xmax1,ymax1|xmin2,ymin2,xmax2,ymax2'.
    - airport_iatas (str): Filtering by Airport IATA codes. Codes should be listed in comma separated string. Example: 'LAX,JFK'.
    - country_names (str): Filtering by full country names. Country names should be listed in comma separated string. Example: 'United States,Canada'.
    - min_capacity (int): Filtering by minimum airport capacity.
    - max_capacity (int): Filtering by maximum airport capacity.
    - top_n (int): Return top n airports by capacity.
    - bottom_n (int): Return bottom n airports by capacity.
    - airport_names (str): Filtering by full airport names. Airport names should be listed in comma separated string. Example: 'Los Angeles International,John F Kennedy International'].
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query_parts = []
    
    if 'airport_iatas' in kwargs:
        iatas = kwargs['airport_iatas'].split(',') 
        iata_tuples = ','.join(cursor.mogrify("%s", (iata,)).decode('utf-8') for iata in iatas)
        query_parts.append(f"orig in ({iata_tuples})")
        
    if 'country_names' in kwargs:
        country_names = kwargs['country_names'].split(',')
        country_tuples = ','.join(cursor.mogrify("%s", (country,)).decode('utf-8') for country in country_names)
        query_parts.append(f"countryname in ({country_tuples})")
        
    if 'airport_names' in kwargs:
        airport_names = kwargs['airport_names'].split(',')
        airportname_tuples = ','.join(cursor.mogrify("%s", (airport,)).decode('utf-8') for airport in airport_names)
        query_parts.append(f"name in ({airportname_tuples})")
            
    if 'bboxes' in kwargs:
        bboxes = kwargs['bboxes'].split('|')
        bbox_queries = []
        for bbox in bboxes:
            coords = bbox.split(',')
            if len(coords) == 4:
                bbox_queries.append(f"ST_MakeEnvelope({','.join(coords)}, 4326)")
        union_bbox_query = f"ST_Union(ARRAY[{','.join(bbox_queries)}])"
        query_parts.append(f"ST_Within(airport1latlon, {union_bbox_query})")
            
    if 'min_capacity' in kwargs and isinstance(kwargs['min_capacity'], int):
        query_parts.append(f"totalseats >= {kwargs['min_capacity']}")

    if 'max_capacity' in kwargs and isinstance(kwargs['max_capacity'], int):
        query_parts.append(f"totalseats <= {kwargs['max_capacity']}") 
            
    if len(query_parts) == 0:
        return {"error": "No valid parameters provided"}, 400
    
    query = f"SELECT * FROM airports WHERE {' AND '.join(query_parts)}"
    print(query)
    
    if 'top_n' in kwargs and isinstance(kwargs['top_n'], int):
        query += f" ORDER BY totalseats DESC LIMIT {kwargs['top_n']}"
    elif 'bottom_n' in kwargs and isinstance(kwargs['bottom_n'], int):
        query += f" ORDER BY totalseats ASC LIMIT {kwargs['bottom_n']}"
    
    cursor.execute(query)
    results = cursor.fetchall()
    
    columns = [desc[0] for desc in cursor.description]
    result_list = [dict(zip(columns, result)) for result in results]
    
    cursor.close()
    return_db_connection(conn)

    return jsonify(result_list)

def get_routes_from_chat(**kwargs):
    """
    Fetch routes based on various optional filters input as kwargs.
    
    Optional parameters:
    - airline_iatas (str): Filtering by Airline IATA codes. Codes should be listed in comma separated string. Example: 'AA,DL'.
    - source_and_destination_bboxes (array): List of pairs of source and destination bounding box coordinates in the format [{source: 'src1xmin,src1ymin,src1xmax,src1ymax', destination: 'dest1xmin,dest1ymin,dest1xmax,dest1ymax'}, {source: 'src2xmin,src2ymin,src2xmax,src2ymax', destination: 'dest2xmin,dest2ymin,dest2xmax,dest2ymax'}].
    - source_bboxes (str): List of only source bounding box coordinates in the format 'xmin1,ymin1,xmax1,ymax1|xmin2,ymin2,xmax2,ymax2'.
    - destination_bboxes (str): List of only destination bounding box coordinates in the format 'xmin1,ymin1,xmax1,ymax1|xmin2,ymin2,xmax2,ymax2'.
    - source_and_destination_country_names (array): Filtering by source and destination country names. List of pairs of source and destination country names. Example: [{source: 'United States', destination: 'Canada'}, {source: 'Japan', destination: 'Singapore'}]'.
    - source_country_names (str): Filtering by only source country names. Country names should be listed in comma separated string. Example: 'United States,Canada'.
    - destination_country_names (str): Filtering by only destination country names. Country names should be listed in comma separated string. Example: 'United States,Canada'.
    - source_and_destination_airport_iatas (array): Filtering by source and destination airport iata codes. List of pairs of source and destination airport iatas. Example: [{source: 'LAX', destination: 'JFK'}, {source: 'AKL', destination: 'ZQN'}].
    - source_airport_iatas (str): Filtering by only source airport iata codes. Codes should be listed in comma separated string. Example: 'LAX,JFK'.
    - destination_airport_iatas (str): Filtering by only destination airport iata codes. Codes should be listed in comma separated string. Example: 'LAX,JFK'.
    - max_number_of_stops (int): Use to fetch all routes and intermediate routes between 1 or more origins and destinations that include less than or equal to the specified number of stops.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if 'number_of_stops' in kwargs and 'source_airport_iatas' in kwargs and 'destination_airport_iatas' in kwargs:
        number_of_stops = int(kwargs['number_of_stops'])
        # Limit to 3 stops
        if (number_of_stops > 3): 
            number_of_stops = 3
        source_airport_iatas = kwargs['source_airport_iatas'].split(',')
        destination_airport_iatas = kwargs['destination_airport_iatas'].split(',')
        
        paired_airport_iatas = list(zip(source_airport_iatas, destination_airport_iatas))
        
        # values_list = ','.join(cursor.mogrify("(%s,%s)", pair).decode('utf-8') for pair in paired_airport_iatas)
        
        routes_groups = []
        
        for source_iata, destination_iata in paired_airport_iatas:        
            query = f"""
            WITH RECURSIVE route_paths AS (
                -- Base case: start from AKL
                SELECT 
                    r.source_airport, 
                    r.destination_airport, 
                    r.airline, 
                    1 as num_stops,
                    ARRAY[ROW(r.source_airport, a1.templat, a1.templong, a1.airport1latlon), 
                    ROW(r.destination_airport, a2.templat, a2.templong, a2.airport1latlon)] AS path -- keep track of the path with lat/long
                FROM routes r
                JOIN airports a1 ON r.source_airport = a1.orig  -- Join to get source airport details
                JOIN airports a2 ON r.destination_airport = a2.orig  -- Join to get destination airport details
                WHERE r.source_airport = %s
                
                UNION ALL
                
                -- Recursive step: join with the initial routes to find subsequent stops
                SELECT 
                    rp.source_airport, 
                    r.destination_airport, 
                    r.airline, 
                    rp.num_stops + 1 as num_stops,
                    rp.path || ROW(r.destination_airport, a.templat, a.templong, a.airport1latlon) -- append the new destination with lat/long to the path
                FROM route_paths rp
                JOIN routes r ON rp.destination_airport = r.source_airport
                JOIN airports a ON r.destination_airport = a.orig -- Join to get destination airport details
                WHERE rp.num_stops < {number_of_stops} AND r.destination_airport != %s
            )
            -- Final selection where the destination is LED and the number of stops is 3 or less
            SELECT DISTINCT
                source_airport, 
                destination_airport, 
                num_stops, 
                path 
            FROM route_paths
            WHERE destination_airport = %s AND num_stops <= {number_of_stops};
            """         
            
            cursor.execute(query, (source_iata, source_iata, destination_iata))
            results = cursor.fetchall()
            
            columns = [desc[0] for desc in cursor.description]
            result_list = [dict(zip(columns, result)) for result in results]
            
            routes_groups.append(result_list)

        return jsonify(routes_groups)
    
    query = """
    SELECT 
        r.airline,
        r.source_airport,
        src_airport.Name AS source_airport_name,
        src_airport.TotalSeats AS source_airport_total_seats,
        src_airport.Countryname AS source_country_name,
        src_airport.Airport1latlon AS source_airport_geometry,
        src_airport.templat AS source_airport_latitude,
        src_airport.templong AS source_airport_longitude,
        r.destination_airport,
        dest_airport.Name AS destination_airport_name,
        dest_airport.TotalSeats AS destination_airport_total_seats,
        dest_airport.Countryname AS destination_country_name,
        dest_airport.Airport1latlon AS destination_airport_geometry,
        dest_airport.templat AS destination_airport_latitude,
        dest_airport.templong AS destination_airport_longitude
    FROM 
        routes r
    JOIN 
        airports src_airport ON r.source_airport = src_airport.orig
    JOIN 
        airports dest_airport ON r.destination_airport = dest_airport.orig
    WHERE
        1=1
    """
    
    if 'airline_iatas' in kwargs:
        iatas = kwargs['airline_iatas'].split(',') 
        iata_tuples = ','.join(cursor.mogrify("%s", (iata,)).decode('utf-8') for iata in iatas)
        query += f" AND r.airline in ({iata_tuples})"

    if 'source_and_destination_bboxes' in kwargs:
        bbox_pairs = kwargs['source_and_destination_bboxes']
        bbox_conditions = []
        for bbox_pair in bbox_pairs:
            src_coords = bbox_pair['source'].split(',')
            dest_coords = bbox_pair['destination'].split(',')
            src_condition = f"(src_airport.templat BETWEEN {src_coords[1]} AND {src_coords[3]} AND src_airport.templong BETWEEN {src_coords[0]} AND {src_coords[2]})"
            dest_condition = f"(dest_airport.templat BETWEEN {dest_coords[1]} AND {dest_coords[3]} AND dest_airport.templong BETWEEN {dest_coords[0]} AND {dest_coords[2]})"
            bbox_conditions.append(f"({src_condition} AND {dest_condition})")
        query += " AND (" + " OR ".join(bbox_conditions) + ")"

    if 'source_bboxes' in kwargs:
        source_bboxes = kwargs['source_bboxes'].split('|')
        src_bbox_conditions = []
        for bbox in source_bboxes:
            xmin, ymin, xmax, ymax = bbox.split(',')
            src_bbox_conditions.append(f"(src_airport.templat BETWEEN {ymin} AND {ymax} AND src_airport.templong BETWEEN {xmin} AND {xmax})")
        query += " AND (" + " OR ".join(src_bbox_conditions) + ")"

    if 'destination_bboxes' in kwargs:
        destination_bboxes = kwargs['destination_bboxes'].split('|')
        dest_bbox_conditions = []
        for bbox in destination_bboxes:
            xmin, ymin, xmax, ymax = bbox.split(',')
            dest_bbox_conditions.append(f"(dest_airport.templat BETWEEN {ymin} AND {ymax} AND dest_airport.templong BETWEEN {xmin} AND {xmax})")
        query += " AND (" + " OR ".join(dest_bbox_conditions) + ")"

    if 'source_and_destination_country_names' in kwargs:
        country_pairs = kwargs['source_and_destination_country_names']
        country_conditions = []
        for pair in country_pairs:
            src_country = pair['source']
            dest_country = pair['destination']
            country_conditions.append(f"(src_airport.Countryname = {cursor.mogrify('%s', (src_country,)).decode('utf-8')} AND dest_airport.Countryname = {cursor.mogrify('%s', (dest_country,)).decode('utf-8')})")
        query += " AND (" + " OR ".join(country_conditions) + ")"

    if 'source_country_names' in kwargs:
        source_countries = kwargs['source_country_names'].split(',')
        source_country_conditions = [f"src_airport.Countryname = {cursor.mogrify('%s', (country,)).decode('utf-8')}" for country in source_countries]
        query += " AND (" + " OR ".join(source_country_conditions) + ")"

    if 'destination_country_names' in kwargs:
        destination_countries = kwargs['destination_country_names'].split(',')
        destination_country_conditions = [f"dest_airport.Countryname = {cursor.mogrify('%s', (country,)).decode('utf-8')}" for country in destination_countries]
        query += " AND (" + " OR ".join(destination_country_conditions) + ")"

    if 'source_and_destination_airport_iatas' in kwargs:
        iata_pairs = kwargs['source_and_destination_airport_iatas']
        iata_conditions = []
        for pair in iata_pairs:
            src_iata = pair['source']
            dest_iata = pair['destination']
            iata_conditions.append(f"(r.source_airport = {cursor.mogrify('%s', (src_iata,)).decode('utf-8')} AND r.destination_airport = {cursor.mogrify('%s', (dest_iata,)).decode('utf-8')})")
        query += " AND (" + " OR ".join(iata_conditions) + ")"

    if 'source_airport_iatas' in kwargs:
        source_iatas = kwargs['source_airport_iatas'].split(',')
        source_iata_conditions = [f"r.source_airport = {cursor.mogrify('%s', (iata,)).decode('utf-8')}" for iata in source_iatas]
        query += " AND (" + " OR ".join(source_iata_conditions) + ")"

    if 'destination_airport_iatas' in kwargs:
        destination_iatas = kwargs['destination_airport_iatas'].split(',')
        destination_iata_conditions = [f"r.destination_airport = {cursor.mogrify('%s', (iata,)).decode('utf-8')}" for iata in destination_iatas]
        query += " AND (" + " OR ".join(destination_iata_conditions) + ")" 
    
    print(query)     
    cursor.execute(query)
    results = cursor.fetchall()
    
    columns = [desc[0] for desc in cursor.description]
    result_list = [dict(zip(columns, result)) for result in results]
    
    cursor.close()
    return_db_connection(conn)

    return jsonify(result_list)