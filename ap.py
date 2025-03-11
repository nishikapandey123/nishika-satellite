



from flask import Flask, render_template_string, request, Response, url_for
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


    return render_template_string('''
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NDVI Dashboard</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Segoe UI', 'Poppins', sans-serif;
            background-color: #f0f2f5;
            color: #252525;
            overflow: hidden;
            height: 100vh;
            width: 100vw;
            padding: 16px;
        }

        .dashboard-container {
            display: grid;
            grid-template-columns: repeat(12, 1fr);
            grid-template-rows: auto 1fr 1fr;
            gap: 16px;
            height: calc(100vh - 32px);
            width: 100%;
        }

        .top-input-section {
            grid-column: span 12;
            background: linear-gradient(90deg, #2B5876, #4E4376);
            padding: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .top-input-section form {
            display: flex;
            align-items: center;
            gap: 15px;
            flex-wrap: nowrap;
        }

        .input-group {
            display: flex;
            align-items: center;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 6px;
            padding: 0 10px;
        }

        .top-input-section label {
            font-size: 14px;
            color: white;
            margin-right: 5px;
            white-space: nowrap;
        }

        .top-input-section input {
            padding: 10px;
            border: none;
            border-radius: 6px;
            outline: none;
            font-size: 14px;
            background: transparent;
            color: white;
            width: 150px;
        }

        .top-input-section button {
            background: #ff9800;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: bold;
            cursor: pointer;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            white-space: nowrap;
        }

        .top-input-section button:hover {
            background: #e68900;
        }

        .dashboard-card {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .card-header {
            padding: 12px 16px;
            font-size: 16px;
            font-weight: 600;
            color: #333;
            background: #b0a3bd;
            border-bottom: 1px solid #eee;
            display: flex;
            align-items: center;
        }

        .card-header .icon {
            margin-right: 8px;
            font-size: 18px;
        }

        .card-content {
            flex: 1;
            overflow: hidden;
            padding: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
       
        .map-section {
            grid-column: span 6;
            grid-row: span 2;
        }
        
        .ndvi-section {
            grid-column: span 3;
            grid-row: span 1;
        }
        
        .pest-chart-section {
            grid-column: span 3;
            grid-row: span 1;
        }
        
        .country-chart-section {
            grid-column: span 3;
            grid-row: span 1;
        }
        
        .product-chart-section {
            grid-column: span 3;
            grid-row: span 1;
        }
        
        
        .card-content img,
        .card-content svg {
            max-width: 100%;
            max-height: 100%;
            width: auto;
            height: auto;
            object-fit: contain;
        }
        
        
        .map-content {
            width: 100%;
            height: 100%;
            border: none;
            display: block;
        }
        
        
        .custom-scrollbar {
            overflow: auto;
        }
        
        .custom-scrollbar::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        
        .custom-scrollbar::-webkit-scrollbar-track {
            background: #f1f1f1;
        }
        
        .custom-scrollbar::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 3px;
        }
        
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
    </style>
</head>
<body>
    <div class="dashboard-container">
        <!-- Top Input Section -->
        <div class="top-input-section">
            <form method="POST" id="ndvi-form">
                <div class="input-group">
                    <label for="start-date">Start Date:</label>
                    <input type="date" id="start-date" name="start_date">
                </div>
                <div class="input-group">
                    <label for="end-date">End Date:</label>
                    <input type="date" id="end-date" name="end_date">
                </div>
                <div class="input-group">
                    <label for="latitude">Latitude:</label>
                    <input type="text" id="latitude" name="latitude" placeholder="Latitude will auto-fill" readonly>
                </div>
                <div class="input-group">
                    <label for="longitude">Longitude:</label>
                    <input type="text" id="longitude" name="longitude" placeholder="Longitude will auto-fill" readonly>
                </div>
                <button type="submit">🌿 CLEAR </button>
            </form>
        </div>
        
        
        <!-- Map Section - Left side, spanning 2 rows -->
        <div class="dashboard-card map-section">
            <div class="card-header">
                <span class="icon">🌍</span> Interactive Map
            </div>
            <div class="card-content">
                {{ map_html|safe }}
            </div>
        </div>
        
        <!-- NDVI Image Section - Top-Right -->
        <div class="dashboard-card ndvi-section">
            <div class="card-header">
                <span class="icon">📊</span> NDVI Analysis
            </div>
            <div class="card-content">
                {% if ndvi_available %}
                <img src="{{ url_for('get_ndvi_image', lat=lat, lon=lon) }}" alt="NDVI Image">
                {% else %}
                <div class="placeholder-message">Select a point on the map to calculate NDVI</div>
                {% endif %}
            </div>
        </div>
        
        <!-- Pest Chart Section - Top-Right, next to NDVI -->
        <div class="dashboard-card pest-chart-section">
            <div class="card-header">
                <span class="icon">🐛</span> Pest Detection Analysis
            </div>
            <div class="card-content">
                <div id="pest-chart"></div>
            </div>
        </div>
        
        <!-- Country Chart Section - Bottom-Right -->
        <div class="dashboard-card country-chart-section">
            <div class="card-header">
                <span class="icon">📊</span> Data Points Per Country
            </div>
            <div class="card-content">
                <div id="bar-chart"></div>
            </div>
        </div>
        
        <!-- Product Chart Section - Bottom-Right, next to Country Chart -->
        <div class="dashboard-card product-chart-section">
            <div class="card-header">
                <span class="icon">🍌</span> Data Points Per Product Type
            </div>
            <div class="card-content custom-scrollbar">
                <div id="product-chart"></div>
            </div>
        </div>
    </div>

    <!-- Load D3.js -->
    <script src="https://d3js.org/d3.v6.min.js"></script>
    <script>
        // Function to resize charts to fit container
        function resizeCharts() {
            // You would call this function when the page loads and on window resize
            renderCountryChart();
            renderProductChart();
            renderPestChart();
        }
        
        // Country Chart
        function renderCountryChart() {
            // Clear previous chart if any
            d3.select("#bar-chart").html("");
            
            // Data received from Python
            var countryData = {{ country_data_json|safe }};
            
            // Get dimensions of the container
            var container = document.querySelector(".country-chart-section .card-content");
            var containerWidth = container.clientWidth;
            var containerHeight = container.clientHeight;
            
            // Set dimensions
            var margin = { top: 20, right: 30, bottom: 40, left: 50 },
                width = containerWidth - margin.left - margin.right,
                height = containerHeight - margin.top - margin.bottom;
                
            // Create SVG container
            var svg = d3.select("#bar-chart")
                .append("svg")
                .attr("width", containerWidth)
                .attr("height", containerHeight)
                .append("g")
                .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
                
            // X scale
            var x = d3.scaleBand()
                .domain(countryData.map(d => d.country))
                .range([0, width])
                .padding(0.3);
                
            // Y scale
            var y = d3.scaleLinear()
                .domain([0, d3.max(countryData, d => d.count)])
                .nice()
                .range([height, 0]);
                
            // Add bars
            svg.selectAll(".bar")
                .data(countryData)
                .enter()
                .append("rect")
                .attr("class", "bar")
                .attr("x", d => x(d.country))
                .attr("y", d => y(d.count))
                .attr("width", x.bandwidth())
                .attr("height", d => height - y(d.count))
                .attr("fill", "#ff9800")
                .on("mouseover", function () { d3.select(this).attr("fill", "#e68900"); })
                .on("mouseout", function () { d3.select(this).attr("fill", "#ff9800"); });
                
            // X-axis
            svg.append("g")
                .attr("transform", "translate(0," + height + ")")
                .call(d3.axisBottom(x))
                .selectAll("text")
                .style("font-size", "10px");
                
            // Y-axis
            svg.append("g")
                .call(d3.axisLeft(y))
                .selectAll("text")
                .style("font-size", "10px");
                
            // Labels
            svg.selectAll(".label")
                .data(countryData)
                .enter()
                .append("text")
                .attr("class", "label")
                .attr("x", d => x(d.country) + x.bandwidth() / 2)
                .attr("y", d => y(d.count) - 5)
                .attr("text-anchor", "middle")
                .attr("fill", "#333")
                .attr("font-size", "10px")
                .text(d => d.count);
        }
        
        // Product Chart
        function renderProductChart() {
            // Clear previous chart if any
            d3.select("#product-chart").html("");
            
            var productData = {{ product_data_json|safe }};
            
            // Get dimensions of the container
            var container = document.querySelector(".product-chart-section .card-content");
            var containerWidth = container.clientWidth;
            var containerHeight = container.clientHeight;
            
            // If no valid product data, show a message instead of an empty chart
            if (productData.length === 0 || (productData.length === 1 && productData[0].product === "No Data")) {
                d3.select("#product-chart").append("p")
                    .text("No product data available")
                    .style("color", "#666")
                    .style("text-align", "center")
                    .style("font-size", "14px");
            } else {
                // Set dimensions
                var productMargin = { top: 20, right: 30, bottom: 60, left: 50 },
                    productWidth = containerWidth - productMargin.left - productMargin.right,
                    productHeight = containerHeight - productMargin.top - productMargin.bottom;
                    
                // Create SVG container
                var productSvg = d3.select("#product-chart")
                    .append("svg")
                    .attr("width", containerWidth)
                    .attr("height", containerHeight)
                    .append("g")
                    .attr("transform", "translate(" + productMargin.left + "," + productMargin.top + ")");
                    
                // X scale
                var productX = d3.scaleBand()
                    .domain(productData.map(d => d.product))
                    .range([0, productWidth])
                    .padding(0.2);
                    
                // Y scale
                var productY = d3.scaleLinear()
                    .domain([0, d3.max(productData, d => d.count)])
                    .nice()
                    .range([productHeight, 0]);
                    
                // Add bars
                productSvg.selectAll(".bar-product")
                    .data(productData)
                    .enter()
                    .append("rect")
                    .attr("class", "bar-product")
                    .attr("x", d => productX(d.product))
                    .attr("y", d => productY(d.count))
                    .attr("width", productX.bandwidth())
                    .attr("height", d => productHeight - productY(d.count))
                    .attr("fill", "#4CAF50")
                    .on("mouseover", function () { d3.select(this).attr("fill", "#388E3C"); })
                    .on("mouseout", function () { d3.select(this).attr("fill", "#4CAF50"); });
                    
                // X-axis with rotated text for readability
                productSvg.append("g")
                    .attr("transform", "translate(0," + productHeight + ")")
                    .call(d3.axisBottom(productX))
                    .selectAll("text")
                    .style("text-anchor", "end")
                    .attr("dx", "-.8em")
                    .attr("dy", ".15em")
                    .attr("transform", "rotate(-30)")
                    .style("font-size", "10px");
                    
                // Y-axis
                productSvg.append("g")
                    .call(d3.axisLeft(productY))
                    .selectAll("text")
                    .style("font-size", "10px");
                    
                // Labels
                productSvg.selectAll(".label-product")
                    .data(productData)
                    .enter()
                    .append("text")
                    .attr("class", "label-product")
                    .attr("x", d => productX(d.product) + productX.bandwidth() / 2)
                    .attr("y", d => productY(d.count) - 5)
                    .attr("text-anchor", "middle")
                    .attr("fill", "#333")
                    .attr("font-size", "10px")
                    .text(d => d.count);
            }
        }
        
        // Pest Chart
        function renderPestChart() {
            // Clear previous chart if any
            d3.select("#pest-chart").html("");
            
            var pestData = {{ pest_data_json|safe }};
            
            // Get dimensions of the container
            var container = document.querySelector(".pest-chart-section .card-content");
            var containerWidth = container.clientWidth;
            var containerHeight = container.clientHeight;
            
            if (pestData.length === 0) {
                d3.select("#pest-chart").append("p")
                    .text("No pest detection data available")
                    .style("color", "#666")
                    .style("text-align", "center")
                    .style("font-size", "14px");
            } else {
                // Set dimensions
                var pestMargin = { top: 20, right: 30, bottom: 40, left: 60 },
                    pestWidth = containerWidth - pestMargin.left - pestMargin.right,
                    pestHeight = containerHeight - pestMargin.top - pestMargin.bottom;
                    
                var pestSvg = d3.select("#pest-chart")
                    .append("svg")
                    .attr("width", containerWidth)
                    .attr("height", containerHeight)
                    .append("g")
                    .attr("transform", "translate(" + pestMargin.left + "," + pestMargin.top + ")");
                    
                var categories = ["Healthy Area", "Diseased Area"];
                var values = [pestData[0].healthy_area, pestData[0].diseased_area];
                
                var pestX = d3.scaleBand()
                    .domain(categories)
                    .range([0, pestWidth])
                    .padding(0.4);
                    
                var pestY = d3.scaleLinear()
                    .domain([0, 100])
                    .nice()
                    .range([pestHeight, 0]);
                    
                var colorScale = d3.scaleOrdinal()
                    .domain(categories)
                    .range(["#4CAF50", "#FF0000"]);  // Green for Healthy, Red for Diseased
                    
                pestSvg.selectAll(".bar-pest")
                    .data(categories)
                    .enter()
                    .append("rect")
                    .attr("class", "bar-pest")
                    .attr("x", d => pestX(d))
                    .attr("y", (d, i) => pestY(values[i]))
                    .attr("width", pestX.bandwidth())
                    .attr("height", (d, i) => pestHeight - pestY(values[i]))
                    .attr("fill", d => colorScale(d))
                    .on("mouseover", function() { d3.select(this).attr("opacity", 0.8); })
                    .on("mouseout", function(d) { d3.select(this).attr("opacity", 1); });
                    
                pestSvg.append("g")
                    .attr("transform", "translate(0," + pestHeight + ")")
                    .call(d3.axisBottom(pestX))
                    .selectAll("text")
                    .style("font-size", "10px");
                    
                pestSvg.append("g")
                    .call(d3.axisLeft(pestY))
                    .selectAll("text")
                    .style("font-size", "10px");
                    
                // Labels
                pestSvg.selectAll(".label-pest")
                    .data(categories)
                    .enter()
                    .append("text")
                    .attr("class", "label-pest")
                    .attr("x", d => pestX(d) + pestX.bandwidth() / 2)
                    .attr("y", (d, i) => pestY(values[i]) - 5)
                    .attr("text-anchor", "middle")
                    .attr("fill", "#333")
                    .attr("font-size", "10px")
                    .text((d, i) => values[i].toFixed(2) + "%");
            }
        }

        // Fill coordinates function
        function fillCoordinates(lat, lon) {
            document.getElementById("latitude").value = lat;
            document.getElementById("longitude").value = lon;
            document.getElementById("ndvi-form").submit();
        }
        
        // Initialize charts when page loads
        window.addEventListener('load', function() {
            resizeCharts();
        });
        
        // Resize charts when window is resized
        window.addEventListener('resize', function() {
            resizeCharts();
        });
    </script>
</body>
</html>
    
    
    
    ''', map_html=map_html, ndvi_available=ndvi_available, lat=lat, lon=lon, country_data_json=country_data_json, product_data_json=product_data_json, pest_data_json=pest_data_json,
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
