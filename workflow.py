"""
Global Bikeability Benchmarking – workflow.py
========================================================
Generates index values for bikeability, connectivity
and accessibility described in:

M. Hawner et al. (2025): "Global Bikeability Benchmarking:
Comparative Analysis of 100 Cities Worldwide Using NetAScore",
Journal of Cycling & Micromobility Research. DOI: 

Author: Maximilian Hawner, University of Salzburg  
Contact:   
Version:   1.0 – 2025-06-19
License:   

Usage
-----
edit the following:
- working_directory: path to folder where NetAScore tool is stored
- input_csv: path to location where city list is stored
- result_output_folder_path: path to location where you want the output to be stored


Requirements
------------
Python ≥3.9; others: see import

This script expects the list of cities from a CSV located in `input_csv`
and produces bikeability, connectivity, and accessibility metrics as described in Section 2.1
of the paper. The results are then stored in a merged CSV comprising all the data from all the cities.

Copyright © 2025 Maximilian Hawner
"""

# import of libraries
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point
import numpy as np
import yaml
import subprocess
import psycopg2
import pandas as pd
import os
import requests

# additional functions, parameter and file paths ----------------------------
working_directory = r'C:\Users\Maxim\Documents\Studium\Master\Salzburg\Master_Thesis\Netascore'
result_output_folder_path=r"C:\Users\Maxim\Documents\Studium\Master\Salzburg\Master_Thesis\output"
input_csv = r"C:\Users\Maxim\Documents\Studium\Master\Salzburg\Master_Thesis\city_list.csv"


db_parameter = {
        'dbname': 'postgres',
        'user': 'postgres',
        'password': 'postgres',
        'host': 'localhost',
        'port': '5433'
    }


def read_city_names_with_pandas(file_path):
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        df['name'] = df['name'].str.strip()
        df['name_en'] = df['name_en'].str.strip()
        names = df['name'].tolist()
        names_en = df['name_en'].tolist()
        names_center = df['name_center'].tolist()
        return names, names_en, names_center

def format_city_name(city_name):
    return city_name.replace(' ', '')


def update_yaml_file(yaml_file_path, city_name, city_name_en):
    with open(yaml_file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    
    if 'global' in data:
        data['global']['case_id'] = city_name_en
    
    if 'import' in data:
        data['import']['place_name'] = city_name
    
    with open(yaml_file_path, 'w', encoding='utf-8') as file:
        yaml.safe_dump(data, file, default_flow_style=False)

def execute_commands(working_directory):
    command = 'docker-compose run netascore'
            
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=working_directory
    )
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        raise RuntimeError(f"Error executing '{command}':\n{stderr}")
    else:
        print(f"Success in executing '{command}':\n{stdout}")

def load_geopackage_layers(gpk_path):
        edge_gdf = None
        city_center_gdf = None
        nodes_gdf = None
        boundary_gdf = None

        try:
            edge_gdf = gpd.read_file(gpk_path, layer='edge')
        except Exception as e:
            print(f"Error loading 'edge': {e}")

        try:
            city_center_gdf = gpd.read_file(gpk_path, layer='city_center')
        except Exception as e:
            print(f"Error loading 'city_center': {e}")

        try:
            nodes_gdf = gpd.read_file(gpk_path, layer='node')
        except Exception as e:
            print(f"Error loading 'node': {e}")

        try:
            boundary_gdf = gpd.read_file(gpk_path, layer='boundary')
        except Exception as e:
            print(f"Error loading 'boundary': {e}")

        return edge_gdf, city_center_gdf, nodes_gdf, boundary_gdf


def import_boundary(db_parameter, city_name):
    try:
        conn = psycopg2.connect(**db_parameter)
        query = 'SELECT * FROM "netascore_data"."aoi"'
        gdf = gpd.read_postgis(query, conn, geom_col='geom', crs='EPSG:4326')
        conn.close()

        # Filter for the specific city
        filtered_gdf = gdf[gdf['name'] == city_name]
        if filtered_gdf.empty:
            print(f"[Info] No boundary found for city '{city_name}'.")
            return None

        return filtered_gdf

    except Exception as e:
        print(f"[Error] Failed to retrieve boundary for '{city_name}': {e}")
        return None

def calculate_netascore_distribution(edge_gdf, output_csv):
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    thresholds = bins[:-1]
    values = edge_gdf['index_bike_ft'].dropna().values
    total = len(values)
    
    if total == 0:
        print("No valid index values found. Skipping.")
        return

    counts, _ = np.histogram(values, bins=bins)
    percentages = (counts / total * 100).round(2)

    try:
        df_out = pd.read_csv(output_csv)
    except FileNotFoundError:
        df_out = pd.DataFrame(columns=['Threshold', 'percentage_edges_classes'])

    # Ersetze ggf. bestehende Zeilen mit denselben Thresholds
    for thr, pct in zip(thresholds, percentages):
        if thr in df_out['Threshold'].values:
            df_out.loc[df_out['Threshold'] == thr, 'percentage_edges_classes'] = pct
        else:
            df_out = pd.concat([df_out, pd.DataFrame([{
                'Threshold': thr,
                'percentage_edges_classes': pct
            }])], ignore_index=True)

    df_out.to_csv(output_csv, index=False)
    print(f"Updated bikeability distribution in {output_csv}")

def create_graph(gdf, source_col, target_col):
    """
    Creates a NetworkX graph from the GeoDataFrame.
    """
    G = nx.MultiGraph()
    for idx, row in gdf.iterrows():
        G.add_edge(row[source_col], row[target_col])
    return G

def write_city_center_to_gpkg(lat, lon, gpk_path, city_name):
    try:
        gdf = gpd.GeoDataFrame(
            {'city': [city_name]},
            geometry=[Point(lon, lat)],
            crs='EPSG:4326'
        )
        gdf.to_file(gpk_path, layer='city_center', driver='GPKG', mode='a')
        print(f"City center for {city_name} written to GeoPackage: {gpk_path}")
        return gdf
    except Exception as e:
        print(f"Error writing city center to GeoPackage for {city_name}: {e}")
        return None

def get_city_center_and_create_shapefile(gpk_path, city_name_formatted, name_en, name_center):
    query = f"""
    [out:json][timeout:25];
    node["place"="city"]["name"="{name_center}"];
    out;
    """
    overpass_url = "http://overpass-api.de/api/interpreter"
    response = requests.post(overpass_url, data={'data': query})

    if response.status_code == 200:
        data = response.json()
        if 'elements' in data and len(data['elements']) > 0:
            element = data['elements'][0]
            lat = element['lat']
            lon = element['lon']
            return write_city_center_to_gpkg(lat, lon, gpk_path, city_name_formatted)
        else:
            print(f"No city center found for {name_en}")
            return None
    else:
        print(f"Error retrieving data for {name_en}: {response.status_code}")
        return None


# ----------------------------------------------------------------

# bikeability function
def bikeability(working_directory, city_name, city_name_en, db_parameter, thresholds, output_folder, output_csv):
    '''
    1. Usage of NetAScore to: 1) retrieve network (edges and nodes) from OpenStreetMap (OSM), 2) calculate and apply index value on edges, 
    and 3) store the network in a GeoPackage (GPKG).
    2. Clipping of network to administrative boundary. 
    3. Calculation of index value distribution and storing in the output_csv.

    Parameters:
    -----------
    city_name: str
        Name of the city (used for OSM retrieval).
    city_name_en: str
        English version of the city name (used for file naming and identification).
    db_parameter: dict
        Dictionary containing database connection parameters.
    thresholds: list of float
        List of threshold values ([0, 0.2, 0.4, 0.6, 0.8]) used to filter the network by bikeability index.
    output_folder: str
        Path to the folder where intermediate or processed files are stored.
    output_csv: str
        Path to the CSV file where summary results are stored.
    '''

    # 1. Edit yaml file
    city_name_formatted = format_city_name(city_name_en)
    yaml_file_path = os.path.join(working_directory, 'data', 'settings_osm_query.yml')
    update_yaml_file(yaml_file_path, city_name, city_name_formatted)
    # 2. NetAScore execution
    try:
        execute_commands(working_directory)
    except RuntimeError as e:
        raise ValueError(f"NetAScore could not be exexuted for {city_name_en}.\n{str(e)}")
    # 3. Connect to database and extract boundary
    gpkg_path = os.path.join(working_directory, 'data', f'netascore_{city_name_formatted}.gpkg')
    edge_gdf, city_center, node_gdf, boundary_gdf = load_geopackage_layers(gpkg_path)
    boundary_gdf = import_boundary(db_parameter, city_name_formatted)
    # 4. Clip NetAScore results to boundary

    if boundary_gdf is not None:
        # Stelle sicher, dass boundary_gdf im selben CRS wie die Netzwerkelemente ist
        boundary_gdf = boundary_gdf.to_crs(edge_gdf.crs)

        # Clipping der Netzwerkelemente
        clipped_edges = gpd.clip(edge_gdf, boundary_gdf)
        clipped_nodes = gpd.clip(node_gdf, boundary_gdf)
    else:
        print(f"Skipping clipping – no boundary found for '{city_name}'")

    # 5. Store result of clipping and boundary as GPKG in output folder
    gpkg_path_city = os.path.join(output_folder, f'{city_name_formatted}.gpkg')
    clipped_edges.to_file(gpkg_path_city, layer="edge", driver="GPKG", mode = 'a')
    clipped_nodes.to_file(gpkg_path_city, layer="node", driver="GPKG", mode = 'a')
    boundary_gdf.to_file(gpkg_path_city, layer="boundary", driver="GPKG", mode = 'a')
    # 6. Calculate distribution of NetAScore index value using thresholds as class borders and results of distribution calculation in output_csv
    calculate_netascore_distribution(edge_gdf, output_csv)


# connectivity function
def connectivity(city_name_en, thresholds, output_folder, output_csv):
    '''
    1. Filtering of network according to threshold.
    2. Calculation of connectivtiy index values (Alpha, Beta, Gamma Index) on network level and storing in the output_csv.

    Parameters:
    -----------
    city_name_en: str
        English name of the city (used for file naming and identification).
    thresholds: list of float
        List of threshold values ([0, 0.2, 0.4, 0.6, 0.8]) used to filter the network by bikeability index.
    output_folder: str
        Path to the folder where intermediate or processed files are be stored.
    output_csv: str
        Path to the CSV file where summary results are stored.
    '''
    
    # 1. Load edges from GPKG in output_folder
    city_name_formatted = format_city_name(city_name_en)
    gpkg_path_city = os.path.join(output_folder, f'{city_name_formatted}.gpkg')
    edge_gdf, city_center_gdf, node_gdf, boundary_gdf = load_geopackage_layers(gpkg_path_city)
    # 2. for loop through thresholds
    for threshold in thresholds:
        # 2.1 Filter edges according to current threshold
        filtered_edges = edge_gdf[edge_gdf['index_bike_ft'] > threshold]
        # 2.2 Create graph using the edges and 'from_node' and 'to_node' column
        G = create_graph(filtered_edges, 'from_node', 'to_node')
        # 2.3 Determine number of edges, nodes, subgraphs
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        num_subgraphs = nx.number_connected_components(G)     
        # 2.4 Calculate Overall Alpha, Beta, Gamma Index
        alpha = (num_edges - num_nodes + num_subgraphs) / (2 * num_nodes - 5) if (2 * num_nodes - 5) != 0 else np.nan
        beta = num_edges / num_nodes if num_nodes != 0 else np.nan
        gamma = num_edges / (3 * (num_nodes - 2)) if (num_nodes - 2) != 0 else np.nan
        # 2.5 Store metrics in output_csv (number of edges, nodes, subgraphs, Alpha, Beta, Gamma)
        try:
            df_out = pd.read_csv(output_csv)
        except FileNotFoundError:
            raise FileNotFoundError(f"Output CSV not found: {output_csv}")

        if threshold in df_out['Threshold'].values:
            row_index = df_out.index[df_out['Threshold'] == threshold][0]
            df_out.at[row_index, 'Total_Num_Nodes'] = num_nodes
            df_out.at[row_index, 'Total_Num_Edges'] = num_edges
            df_out.at[row_index, 'Total_Num_Subgraphs'] = num_subgraphs
            df_out.at[row_index, 'Overall Alpha-Index'] = alpha
            df_out.at[row_index, 'Overall Beta-Index'] = beta
            df_out.at[row_index, 'Overall Gamma-Index'] = gamma
        else:
            new_row = {
                'Threshold': threshold,
                'Total_Num_Nodes': num_nodes,
                'Total_Num_Edges': num_edges,
                'Total_Num_Subgraphs': num_subgraphs,
                'Overall Alpha-Index': alpha,
                'Overall Beta-Index': beta,
                'Overall Gamma-Index': gamma
            }
            df_out = pd.concat([df_out, pd.DataFrame([new_row])], ignore_index=True)

        df_out.to_csv(output_csv, index=False)

# accessibility function
def accessibility(city_name_en, city_name_center, thresholds, output_folder, output_csv):
    '''
    1. Filtering of network according to threshold.
    2. Calculation of accessibility index value and storing in the output_csv.

    Parameters:
    -----------
    city_name_en: str
        English name of the city (used for file naming and identification).
    city_name_center: str
        Name of the city center (used for OSM retrieval).
    thresholds: list of float
        List of threshold values ([0, 0.2, 0.4, 0.6, 0.8]) used to filter the network by bikeability index.
    output_folder: str
        Path to the folder where intermediate or processed files are be stored.
    output_csv: str
        Path to the CSV file where summary results are stored.
    '''
    
    # 1. Load edges from GPKG in output_folder
    city_name_formatted = format_city_name(city_name_en)
    gpkg_path_city = os.path.join(output_folder, f'{city_name_formatted}.gpkg')
    edge_gdf, city_center_gdf, node_gdf, boundary_gdf = load_geopackage_layers(gpkg_path_city)
    nodes = node_gdf.to_crs(epsg=3857)
    nodes["node_id"] = nodes.index + 1
    edge_gdf = edge_gdf.to_crs(epsg=3857)
    # 2. Create global network and calculate largest connected component (LCC)
    global_network = create_graph(edge_gdf, 'from_node', 'to_node')
    lcc = max(nx.connected_components(global_network), key=len)
    # 3. Calculate number of edges that allow cycling
    rated_network = edge_gdf[edge_gdf['index_bike_ft']>0]
    # 4. Extract city center from OSM and store in GPKG
    city_center_gdf = get_city_center_and_create_shapefile(gpkg_path_city, city_name_formatted, city_name_en, city_name_center)
    city_center = city_center_gdf.to_crs(epsg=3857)
    # 5. for loop through thresholds
    for threshold in thresholds:
        # 5.1 Filter edges according to current threshold
        filtered_edges = edge_gdf[edge_gdf["index_bike_ft"] >= threshold]
        # 5.2 Create graph using the edges and 'from_node' and 'to_node' column
        G = create_graph(filtered_edges, 'from_node', 'to_node')
        # 5.3 Create set of nodes from edges
        filtered_nodes = set(filtered_edges['from_node']).union(set(filtered_edges['to_node']))
        # 5.4 Create 500-meter buffer around city center
        buffer_500 = city_center.geometry.iloc[0].buffer(500)
        # 5.5 Select nodes that are within the buffer AND belong to LCC -> destination nodes
        nodes_in_buffer = nodes[nodes["node_id"].isin(filtered_nodes) & nodes.geometry.within(buffer_500)]
        filtered_nodes_in_buffer = set(nodes_in_buffer["node_id"])
        valid_nodes_set = filtered_nodes_in_buffer.intersection(lcc)
        # 5.6 Identify edges that can be reached from destination nodes
        reachable_edges_set = set()
        visited = set()
        for node in valid_nodes_set:
            if node in visited or node not in G:
                continue

            component_nodes = nx.node_connected_component(G, node)
            visited.update(component_nodes)
            
            for edge in G.subgraph(component_nodes).edges():
                reachable_edges_set.add(tuple(sorted(edge)))
                
        filtered_edges_tuples = filtered_edges.apply(lambda x: tuple(sorted((x['from_node'], x['to_node']))), axis=1)
        reachable_edges_gdf = filtered_edges[filtered_edges_tuples.isin(reachable_edges_set)]
        # 5.7 Calculate share of identified edges in relation to edges where cycling is allowed
        total_edges_rated = len(rated_network)
        reachable_percentage_rated = (len(reachable_edges_gdf) / total_edges_rated) * 100 if total_edges_rated > 0 else 0
        # 5.8 Store metric in output_csv
        results_df = pd.read_csv(output_csv)
        results_df.loc[results_df['Threshold'] == float(threshold), 'reachable_percentage'] = reachable_percentage_rated
        results_df.to_csv(output_csv, index=False)

# main function
if __name__ == "__main__":

    # 1. Define thresholds
    thresholds=[0, 0.2, 0.4, 0.6, 0.8]

    # 2. Read city names out of input_csv
    names, names_en, names_center = read_city_names_with_pandas(input_csv)

    # 3. for loop through input_csv
    for i in range(len(names_en)):
        # 3.1 Extract and format city names: name, name_en, name_center
        name = names[i]
        name_en = names_en[i]
        name_center = names_center[i]
        city_name_formatted = format_city_name(name_en)
        # 3.2 Create output folder
        output_folder = os.path.join(result_output_folder_path, city_name_formatted)
        os.makedirs(output_folder, exist_ok=True)
        # 3.3 Create output CSV in output folder (columns: Threshold, Total_Num_Nodes, Total_Num_Edges, Total_Num_Subgraphs, percentage_edges_classes, Overall Alpha-Index, Overall Beta-Index, Overall Gamma-Index, reachable_percentage)
        output_csv = os.path.join(output_folder, f"{name_en}_network_metrics.csv")
        header = [
            "Threshold",
            "Total_Num_Nodes",
            "Total_Num_Edges",
            "Total_Num_Subgraphs",
            "percentage_edges_classes",
            "Overall Alpha-Index",
            "Overall Beta-Index",
            "Overall Gamma-Index",
            "reachable_percentage"
            ]

        with open(output_csv, mode='w', encoding='utf-8') as f:
            f.write(','.join(header) + '\n')
        # 3.4 Call bikeability function (name, name_en, db_parameter, thresholds, output_folder, output_csv)
        print(f"Bikeability calculation for {name_en} started")
        bikeability(working_directory, name, name_en, db_parameter, thresholds, output_folder, output_csv)
        print(f"Bikeability calculation finished.")
        # 3.5 Call connectivity function (name_en, thresholds, output_folder, output_csv)
        print(f"Connectivity calculation for {name_en} started")
        connectivity(name_en, thresholds, output_folder, output_csv)
        print(f"Connectivity calculation finished.")
        # 3.6 Call accessibility function (name_en, name_center, thresholds, output_folder, output_csv)
        print(f"Accessibility calculation for {name_en} started")
        accessibility(name_en, name_center, thresholds, output_folder, output_csv)
        print(f"Accessibility calculation finished.")
    # 4. Create csv for merging the data for all the cities together (columns: City, Threshold, Total_Num_Nodes, Total_Num_Edges, Total_Num_Subgraphs, percentage_edges_classes, Overall Alpha-Index, Overall Beta-Index, Overall Gamma-Index, reachable_percentage)
    merged_csv_path = os.path.join(result_output_folder_path, "merged_results.csv")
    merged_header = [
        "City",
        "Threshold",
        "Total_Num_Nodes",
        "Total_Num_Edges",
        "Total_Num_Subgraphs",
        "percentage_edges_classes",
        "Overall Alpha-Index",
        "Overall Beta-Index",
        "Overall Gamma-Index",
        "reachable_percentage"
    ]
    if not os.path.exists(merged_csv_path):
        with open(merged_csv_path, mode='w', encoding='utf-8') as f:
            f.write(','.join(merged_header) + '\n')
    # 5. for loop through input_csv
    for i in range(len(names_en)):
        # 5.1 Use city_name_en to loate output_csv for each city
        name_en = names_en[i]
        city_output_csv = os.path.join(result_output_folder_path, name_en, f"{name_en}_network_metrics.csv")
        # 5.2 Append metrics of each city to csv
        if os.path.exists(city_output_csv):
            city_df = pd.read_csv(city_output_csv)
            city_df.insert(0, "City", name_en)  # Add city name as first column
            city_df.to_csv(merged_csv_path, mode='a', index=False, header=False)