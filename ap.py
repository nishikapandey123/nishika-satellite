


# from flask import Flask, render_template, request, Response, url_for
# import pandas as pd
# import folium
# from folium.plugins import MarkerCluster, FastMarkerCluster 
# import ee
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')  # Use non-GUI backend to avoid errors
# import matplotlib.pyplot as plt
# from PIL import Image
# import geemap
# import io

# app = Flask(__name__)

# # Initialize Google Earth Engine (GEE)
# try:
#     ee.Initialize(project="814405731292")
# except:
#     ee.Authenticate()
#     ee.Initialize()

# # Backend: CSV Files (For Map Markers)
# csv_files = {
#     "Colombia": r"data\COLOMBIA - Sheet1 (2).csv",
#     "Peru": r"data\FOC_PERÚ.csv",
#     "Ecuador": r"data\Untitled spreadsheet - FOC_ECUADOR copy.csv",
#     "Bolivia": r"data\FOC_BOLIVIA (2).csv"
# }


# # Load CSV Data for Map
# def load_data():
#     dataframes = []
#     for country, path in csv_files.items():
#         df = pd.read_csv(path)
#         df["country"] = country
#         dataframes.append(df)
#     return pd.concat(dataframes, ignore_index=True)

# df = load_data()
# df = df.dropna(subset=["LATITUD", "LONGITUD"])  # Remove NaN values

# # Function to create the Folium map
# def create_map():
#     # Create base map
#     m = folium.Map(location=[-10, -70], zoom_start=4)

#     # Define unique colors for each country
#     country_colors = {
#         "Colombia": "blue",
#         "Peru": "red",
#         "Ecuador": "green",
#         "Bolivia": "pink"
#     }

#     # Load GeoJSON file for country boundaries
#     geojson_path = "countries.geojson"
#     try:
#         import json
#         with open(geojson_path, "r", encoding="utf-8") as file:
#             geojson_data = json.load(file)

#         selected_countries = ["Colombia", "Peru", "Ecuador", "Bolivia"]
#         geojson_data["features"] = [
#             feature for feature in geojson_data["features"]
#             if feature["properties"].get("name") in selected_countries
#         ]

#         folium.GeoJson(
#             geojson_data,
#             name="Country Boundaries",
#             style_function=lambda x: {
#                 "fillColor": "yellow",
#                 "color": "black",
#                 "weight": 2,
#                 "fillOpacity": 0.3
#             }
#         ).add_to(m)

#     except Exception as e:
#         print(f"Error loading GeoJSON: {e}")

#     # ✅ Use different colors per country
#     for _, row in df.iterrows():
#         country = row["country"]
#         color = country_colors.get(country, "gray")  # Default to gray if country not found

#         folium.CircleMarker(
#             location=[row["LATITUD"], row["LONGITUD"]],
#             radius=3,
#             color=color,
#             fill=True,
#             fill_color=color,
#             fill_opacity=0.7,
#             popup=folium.Popup(f"""
#                 <b>{country}</b><br>
#                 Lat: {row["LATITUD"]}<br>Long: {row["LONGITUD"]}<br>
#                 <button onclick="window.parent.fillCoordinates({row["LATITUD"]}, {row["LONGITUD"]})">
#                     Select
#                 </button>
#             """, max_width=250),
#         ).add_to(m)

#     return m._repr_html_()




# # Function to extract and display NDVI without saving as PNG
# import cv2
# import os
# import json

# # Global dictionary to store pest density data
# pest_data_dict = {}

# # Ensure the directory for storing pest images exists
# os.makedirs("static/pest_images", exist_ok=True)

# def generate_ndvi_plot(lat, lon, start_date="2021-01-01", end_date="2021-12-31"):
#     try:
#         point = ee.Geometry.Point(lon, lat)

#         # Fetch Sentinel-2 imagery
#         data = ee.ImageCollection("COPERNICUS/S2").filterBounds(point)
#         image = ee.Image(data.filterDate(start_date, end_date).sort("CLOUD_COVERAGE_ASSESSMENT").first())

#         # NDVI Calculation
#         NDVI = image.expression(
#             "(NIR - RED) / (NIR + RED)",
#             {
#                 'NIR': image.select("B8"),
#                 'RED': image.select("B4")
#             }
#         )

#         # Scale NDVI for visualization
#         NDVI_scaled = NDVI.multiply(255).toByte()

#         # Clip NDVI around the selected point
#         region = point.buffer(1000).bounds()
#         url = NDVI_scaled.clip(region).getDownloadURL({
#             'scale': 10,
#             'region': region,
#             'format': 'GeoTIFF'
#         })

#         # Download NDVI image and convert to NumPy array
#         try:
#             image_pil = Image.open(geemap.download_file(url)).convert("L")  # Convert to grayscale
#             image_np = np.array(image_pil)
#         except Exception as e:
#             print(f"⚠️ NDVI Image Download Failed: {e}")
#             return None, None, None

#         if image_np is None or image_np.size == 0:
#             raise ValueError("NDVI image could not be processed.")

#         # ---------------------- NDVI VISUALIZATION ---------------------- #
#         fig, ax = plt.subplots(figsize=(6, 5))
#         img_plot = ax.imshow(image_np, cmap='RdYlGn')  # Red-Yellow-Green colormap
#         cbar = plt.colorbar(img_plot, ax=ax)
#         cbar.set_label("NDVI Value")
#         ax.axis("off")
#         ax.set_title(f"NDVI at Lat: {lat}, Lon: {lon}")

#         # Convert Matplotlib figure to in-memory image
#         img_bytes = io.BytesIO()
#         plt.savefig(img_bytes, format="png", bbox_inches="tight")
#         plt.close(fig)
#         img_bytes.seek(0)

#         # ---------------------- PEST DETECTION ---------------------- #
#         # Apply Canny Edge Detection
#         edges = cv2.Canny(image_np, threshold1=50, threshold2=150)

#         # Apply Laplacian (Delight Filter)
#         laplacian = cv2.Laplacian(image_np, cv2.CV_64F)
#         laplacian = np.uint8(np.absolute(laplacian))

#         # Combine Edge Detection & Laplacian for Pest Detection
#         pest_detection = cv2.addWeighted(edges, 0.7, laplacian, 0.3, 0)

#         # Calculate Pest Affected Percentage
#         total_pixels = pest_detection.size
#         diseased_pixels = np.sum(pest_detection > 100)
#         pest_density = (diseased_pixels / total_pixels) * 100
#         healthy_area = 100 - pest_density

#         # Categorize Pest Infection
#         if pest_density < 10:
#             status = "Healthy"
#             color = "green"
#         elif pest_density < 30:
#             status = "Moderate"
#             color = "yellow"
#         else:
#             status = "Diseased"
#             color = "red"

#         # Save Pest Detection Image
#         pest_image_path = f"static/pest_images/pest_{lat}_{lon}.png"
#         cv2.imwrite(pest_image_path, pest_detection)

#         # Store Pest Data for Visualization (Unique for each coordinate)
#         pest_data_dict[f"{lat},{lon}"] = {
#             "lat": lat, "lon": lon,
#             "diseased_area": round(pest_density, 2),
#             "healthy_area": round(healthy_area, 2),
#             "color": color
#         }

#         return img_bytes, pest_image_path, pest_data_dict[f"{lat},{lon}"]

#     except Exception as e:
#         print(f"❌ Error in NDVI & Pest Detection Calculation: {e}")
#         return None, None, None







# # Flask Route for Home
# @app.route("/index.html", methods=["GET", "POST"])
# def index():
#     ndvi_available = False
#     lat, lon, start_date, end_date = None, None, None, None
#     pest_image_path = None
#     pest_data = None  # Store current pest data

#     if request.method == "POST":
#         try:
#             lat = float(request.form["latitude"])
#             lon = float(request.form["longitude"])
#             ndvi_available = True
#             start_date = request.form["start_date"]
#             end_date = request.form["end_date"]
#             _, pest_image_path, pest_data = generate_ndvi_plot(lat, lon, start_date, end_date)

#         except ValueError:
#             pass  # Ignore invalid input

#     map_html = create_map()

#     # Count data points per country
#     country_counts = df["country"].value_counts().to_dict()

#     # Identify the correct column name for product type
#     possible_product_columns = ["PRODUCTO/CULTIVO", "Producto", "PRODUCTO"]
#     product_column = next((col for col in possible_product_columns if col in df.columns), None)

#     # Count data points per product type
#     if product_column and not df[product_column].isnull().all():
#         product_counts = df[product_column].value_counts().to_dict()
#     else:
#         product_counts = {"No Data": 1}  # Avoid empty dataset issue

#     # Convert data to JSON for D3.js visualization
#     import json
#     country_data_json = json.dumps([{"country": k, "count": v} for k, v in country_counts.items()])
#     product_data_json = json.dumps([{"product": k, "count": v} for k, v in product_counts.items()])
    
#     # Pass only the current pest detection data
#     pest_data_json = json.dumps([pest_data]) if pest_data else "[]"


#     return render_template("index.html", map_html=map_html, ndvi_available=ndvi_available, lat=lat, lon=lon, country_data_json=country_data_json, product_data_json=product_data_json, pest_data_json=pest_data_json,
#                            pest_image_path=pest_image_path)



# # Route to generate NDVI dynamically without saving it
# @app.route("/ndvi_image/<lat>/<lon>")
# def get_ndvi_image(lat, lon):
#     lat, lon = float(lat), float(lon)
#     ndvi_image, _, _ = generate_ndvi_plot(lat, lon)  # Correctly unpack 3 values

#     if ndvi_image:
#         return Response(ndvi_image.getvalue(), mimetype="image/png")

#     return "No NDVI Image Available", 404


# @app.route("/pest_image/<lat>/<lon>")
# def get_pest_image(lat, lon):
#     pest_image_path = f"static/pest_images/pest_{lat}_{lon}.png"

#     if os.path.exists(pest_image_path):
#         return Response(open(pest_image_path, "rb").read(), mimetype="image/png")
    
#     return "No Pest Detection Image Available", 404



# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)






from flask import Flask, render_template, request, Response, url_for
import pandas as pd
import folium
from folium.plugins import MarkerCluster, FastMarkerCluster 
import ee
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid errors
import matplotlib.pyplot as plt
from PIL import Image
import geemap
import io
import os

app = Flask(__name__)

# Initialize Google Earth Engine (GEE)
SERVICE_ACCOUNT_PATH = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'proven-space-452610-g1-beef75df7b84.json')
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = SERVICE_ACCOUNT_PATH

try:
    credentials = ee.ServiceAccountCredentials(None, SERVICE_ACCOUNT_PATH)
    ee.Initialize(credentials)
    print("✅ Google Earth Engine authenticated successfully!")
except Exception as e:
    print(f"❌ Error initializing Google Earth Engine: {e}")

# Backend: CSV Files (For Map Markers)
csv_files = {
    "Colombia": r"COLOMBIA - Sheet1 (2).csv",
    "Peru": r"FOC_PERÚ.csv",
    "Ecuador": r"Untitled spreadsheet - FOC_ECUADOR copy.csv",
    "Bolivia": r"FOC_BOLIVIA (2).csv"
}


# Load CSV Data for Map
def load_data():
    dataframes = []
    for country, path in csv_files.items():
        df = pd.read_csv(path)
        df["country"] = country
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

df = load_data()
df = df.dropna(subset=["LATITUD", "LONGITUD"])  # Remove NaN values

# Function to create the Folium map
def create_map():
    # Create base map
    m = folium.Map(location=[-10, -70], zoom_start=4)

    # Define unique colors for each country
    country_colors = {
        "Colombia": "blue",
        "Peru": "red",
        "Ecuador": "green",
        "Bolivia": "pink"
    }

    # Load GeoJSON file for country boundaries
    geojson_path = "countries.geojson"
    try:
        import json
        with open(geojson_path, "r", encoding="utf-8") as file:
            geojson_data = json.load(file)

        selected_countries = ["Colombia", "Peru", "Ecuador", "Bolivia"]
        geojson_data["features"] = [
            feature for feature in geojson_data["features"]
            if feature["properties"].get("name") in selected_countries
        ]

        folium.GeoJson(
            geojson_data,
            name="Country Boundaries",
            style_function=lambda x: {
                "fillColor": "yellow",
                "color": "black",
                "weight": 2,
                "fillOpacity": 0.3
            }
        ).add_to(m)

    except Exception as e:
        print(f"Error loading GeoJSON: {e}")

    # ✅ Use different colors per country
    for _, row in df.iterrows():
        country = row["country"]
        color = country_colors.get(country, "gray")  # Default to gray if country not found

        folium.CircleMarker(
            location=[row["LATITUD"], row["LONGITUD"]],
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(f"""
                <b>{country}</b><br>
                Lat: {row["LATITUD"]}<br>Long: {row["LONGITUD"]}<br>
                <button onclick="window.parent.fillCoordinates({row["LATITUD"]}, {row["LONGITUD"]})">
                    Select
                </button>
            """, max_width=250),
        ).add_to(m)

    return m._repr_html_()




# Function to extract and display NDVI without saving as PNG
import cv2
import os
import json

# Global dictionary to store pest density data
pest_data_dict = {}

# Ensure the directory for storing pest images exists
os.makedirs("static/pest_images", exist_ok=True)

def generate_ndvi_plot(lat, lon, start_date="2021-01-01", end_date="2021-12-31"):
    try:
        point = ee.Geometry.Point(lon, lat)

        # Fetch Sentinel-2 imagery
        data = ee.ImageCollection("COPERNICUS/S2").filterBounds(point)
        image = ee.Image(data.filterDate(start_date, end_date).sort("CLOUD_COVERAGE_ASSESSMENT").first())

        # NDVI Calculation
        NDVI = image.expression(
            "(NIR - RED) / (NIR + RED)",
            {
                'NIR': image.select("B8"),
                'RED': image.select("B4")
            }
        )

        # Scale NDVI for visualization
        NDVI_scaled = NDVI.multiply(255).toByte()

        # Clip NDVI around the selected point
        region = point.buffer(1000).bounds()
        url = NDVI_scaled.clip(region).getDownloadURL({
            'scale': 10,
            'region': region,
            'format': 'GeoTIFF'
        })

        # Download NDVI image and convert to NumPy array
        try:
            image_pil = Image.open(geemap.download_file(url)).convert("L")  # Convert to grayscale
            image_np = np.array(image_pil)
        except Exception as e:
            print(f"⚠️ NDVI Image Download Failed: {e}")
            return None, None, None

        if image_np is None or image_np.size == 0:
            raise ValueError("NDVI image could not be processed.")

        # ---------------------- NDVI VISUALIZATION ---------------------- #
        fig, ax = plt.subplots(figsize=(6, 5))
        img_plot = ax.imshow(image_np, cmap='RdYlGn')  # Red-Yellow-Green colormap
        cbar = plt.colorbar(img_plot, ax=ax)
        cbar.set_label("NDVI Value")
        ax.axis("off")
        ax.set_title(f"NDVI at Lat: {lat}, Lon: {lon}")

        # Convert Matplotlib figure to in-memory image
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format="png", bbox_inches="tight")
        plt.close(fig)
        img_bytes.seek(0)

        # ---------------------- PEST DETECTION ---------------------- #
        # Apply Canny Edge Detection
        edges = cv2.Canny(image_np, threshold1=50, threshold2=150)

        # Apply Laplacian (Delight Filter)
        laplacian = cv2.Laplacian(image_np, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))

        # Combine Edge Detection & Laplacian for Pest Detection
        pest_detection = cv2.addWeighted(edges, 0.7, laplacian, 0.3, 0)

        # Calculate Pest Affected Percentage
        total_pixels = pest_detection.size
        diseased_pixels = np.sum(pest_detection > 100)
        pest_density = (diseased_pixels / total_pixels) * 100
        healthy_area = 100 - pest_density

        # Categorize Pest Infection
        if pest_density < 10:
            status = "Healthy"
            color = "green"
        elif pest_density < 30:
            status = "Moderate"
            color = "yellow"
        else:
            status = "Diseased"
            color = "red"

        # Save Pest Detection Image
        pest_image_path = f"static/pest_images/pest_{lat}_{lon}.png"
        cv2.imwrite(pest_image_path, pest_detection)

        # Store Pest Data for Visualization (Unique for each coordinate)
        pest_data_dict[f"{lat},{lon}"] = {
            "lat": lat, "lon": lon,
            "diseased_area": round(pest_density, 2),
            "healthy_area": round(healthy_area, 2),
            "color": color
        }

        return img_bytes, pest_image_path, pest_data_dict[f"{lat},{lon}"]

    except Exception as e:
        print(f"❌ Error in NDVI & Pest Detection Calculation: {e}")
        return None, None, None







# Flask Route for Home
@app.route("/", methods=["GET", "POST"])
def index():
    ndvi_available = False
    lat, lon, start_date, end_date = None, None, None, None
    pest_image_path = None
    pest_data = None  # Store current pest data

    if request.method == "POST":
        try:
            lat = float(request.form["latitude"])
            lon = float(request.form["longitude"])
            ndvi_available = True
            start_date = request.form["start_date"]
            end_date = request.form["end_date"]
            _, pest_image_path, pest_data = generate_ndvi_plot(lat, lon, start_date, end_date)

        except ValueError:
            pass  # Ignore invalid input

    map_html = create_map()

    # Count data points per country
    country_counts = df["country"].value_counts().to_dict()

    # Identify the correct column name for product type
    possible_product_columns = ["PRODUCTO/CULTIVO", "Producto", "PRODUCTO"]
    product_column = next((col for col in possible_product_columns if col in df.columns), None)

    # Count data points per product type
    if product_column and not df[product_column].isnull().all():
        product_counts = df[product_column].value_counts().to_dict()
    else:
        product_counts = {"No Data": 1}  # Avoid empty dataset issue

    # Convert data to JSON for D3.js visualization
    import json
    country_data_json = json.dumps([{"country": k, "count": v} for k, v in country_counts.items()])
    product_data_json = json.dumps([{"product": k, "count": v} for k, v in product_counts.items()])
    
    # Pass only the current pest detection data
    pest_data_json = json.dumps([pest_data]) if pest_data else "[]"


    return render_template("index.html", map_html=map_html, ndvi_available=ndvi_available, lat=lat, lon=lon, country_data_json=country_data_json, product_data_json=product_data_json, pest_data_json=pest_data_json,
                           pest_image_path=pest_image_path)



# Route to generate NDVI dynamically without saving it
@app.route("/ndvi_image/<lat>/<lon>")
def get_ndvi_image(lat, lon):
    lat, lon = float(lat), float(lon)
    ndvi_image, _, _ = generate_ndvi_plot(lat, lon)  # Correctly unpack 3 values

    if ndvi_image:
        return Response(ndvi_image.getvalue(), mimetype="image/png")

    return "No NDVI Image Available", 404


@app.route("/pest_image/<lat>/<lon>")
def get_pest_image(lat, lon):
    pest_image_path = f"static/pest_images/pest_{lat}_{lon}.png"

    if os.path.exists(pest_image_path):
        return Response(open(pest_image_path, "rb").read(), mimetype="image/png")
    
    return "No Pest Detection Image Available", 404



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
