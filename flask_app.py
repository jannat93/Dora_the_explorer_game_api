# app.py
from flask import Flask, request, jsonify
import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
import os

# ---------------------------
# 1. Load Synthetic Data
# ---------------------------
df = pd.read_csv("chittagong_routes_game.csv")
df.rename(columns={
    "crime_density":"crime",
    "lighting_density":"lighting",
    "crowd_density":"crowd"
}, inplace=True)

# ---------------------------
# 2. Create Ground Truth
# ---------------------------
df["true_label"] = ((df["crime"]>0.6) & (df["lighting"]<0.4) & (df["crowd"]<0.4)).astype(int)

# ---------------------------
# 3. Train Random Forest Model
# ---------------------------
X = df[["crime","lighting","crowd"]]
y = df["true_label"]
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# ---------------------------
# 4. Build Graph for Routes
# ---------------------------
G = nx.Graph()
for _, row in df.iterrows():
    G.add_edge(
        (row.start_lat, row.start_lon),
        (row.end_lat, row.end_lon),
        crime=row.crime,
        lighting=row.lighting,
        crowd=row.crowd,
        distance=row.distance_m
    )

# ---------------------------
# 5. Initialize Flask App
# ---------------------------
app = Flask(__name__)

# ---------------------------
# 6. Risk Score Function
# ---------------------------
def risk_score(edge):
    X_edge = pd.DataFrame([{
        "crime": edge["crime"],
        "lighting": edge["lighting"],
        "crowd": edge["crowd"]
    }])
    pred = rf_model.predict(X_edge)[0]
    return pred

# ---------------------------
# 7. Route Endpoint
# ---------------------------
@app.route("/routes", methods=["POST"])
def get_routes():
    data = request.get_json()
    start = (data["start_lat"], data["start_lon"])
    end = (data["end_lat"], data["end_lon"])
    max_hops = 5  # Limit path length for performance

    try:
        paths = list(nx.all_simple_paths(G, source=start, target=end, cutoff=max_hops))
    except nx.NetworkXNoPath:
        return jsonify({"error": "No path found"}), 404

    route_list = []
    for path in paths:
        total_distance = 0
        unsafe_edges = 0
        for i in range(len(path)-1):
            edge = G.get_edge_data(path[i], path[i+1])
            total_distance += edge["distance"]
            if risk_score(edge):
                unsafe_edges += 1
        route_list.append({
            "path": path,
            "distance": total_distance,
            "unsafe_edges": unsafe_edges,
            "risk_score": unsafe_edges/(len(path)-1) if len(path)>1 else 0
        })

    return jsonify(route_list)

# ---------------------------
# 8. Run App
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render provides PORT
    app.run(host="0.0.0.0", port=port)
