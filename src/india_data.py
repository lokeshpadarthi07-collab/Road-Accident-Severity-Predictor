"""
India Highway Accident Blackspots & Safety Awareness Data Module.
Grounded in Ministry of Road Transport and Highways (MoRTH) accident reports,
Integrated Road Accident Database (iRAD), and Indian Road Congress (IRC) blackspot guidelines.
"""

import os
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Complete Major Indian Highway Network Polylines across India (Deck.gl PathLayer format: [longitude, latitude])
INDIA_HIGHWAY_NETWORKS = [
    {
        "highway_id": "NH-44",
        "highway_name": "NH-44: North-South Trans-India Corridor (Srinagar to Kanyakumari)",
        "short_name": "NH-44",
        "length_km": 4112,
        "category": "National Highway",
        "danger_level": "High Danger",
        "color": [6, 182, 212, 220],       # Glowing Cyan
        "highlight_color": [239, 68, 68, 255], # Red when active
        "total_blackspots": 184,
        "description": "India's longest highway spanning 11 states from Jammu & Kashmir to Tamil Nadu. Critical hazards include heavy winter fog in Punjab/Haryana and night fatigue crashes in central/southern sectors.",
        "path": [
            [74.7973, 34.0837],  # Srinagar
            [74.8570, 32.7266],  # Jammu
            [75.6525, 32.2643],  # Pathankot
            [75.5762, 31.3260],  # Jalandhar
            [75.8920, 30.8980],  # Ludhiana (Sherpur Chowk Blackspot)
            [76.7767, 30.3782],  # Ambala
            [76.9635, 29.6857],  # Karnal
            [77.0150, 28.9930],  # Panipat - Murthal Blackspot Corridor
            [77.1025, 28.7041],  # Delhi Bypass
            [77.3178, 28.4089],  # Faridabad
            [77.6737, 27.4924],  # Mathura
            [78.0081, 27.1767],  # Agra
            [78.1828, 26.2183],  # Gwalior
            [78.5685, 25.4484],  # Jhansi
            [78.7378, 23.8388],  # Sagar
            [79.0882, 21.1458],  # Nagpur
            [78.5320, 19.6641],  # Adilabad
            [78.4867, 17.3850],  # Hyderabad
            [78.0370, 15.8280],  # Kurnool Blackspot
            [77.6006, 14.6819],  # Anantapur
            [77.5946, 12.9716],  # Bengaluru
            [77.8253, 12.7409],  # Hosur
            [78.1460, 11.6643],  # Salem
            [78.1198, 9.9252],   # Madurai
            [77.7567, 8.7139],   # Tirunelveli
            [77.5385, 8.0883]    # Kanyakumari
        ]
    },
    {
        "highway_id": "NH-48",
        "highway_name": "NH-48: Golden Quadrilateral (Delhi - Jaipur - Mumbai - Pune - Bengaluru - Chennai)",
        "short_name": "NH-48",
        "length_km": 2807,
        "category": "National Highway",
        "danger_level": "Critical Danger",
        "color": [245, 158, 11, 220],      # Amber Orange
        "highlight_color": [239, 68, 68, 255],
        "total_blackspots": 218,
        "description": "Busiest economic freight corridor in India connecting major metropolitan economic powerhouses. Severe crash clusters at steep downhill bypasses and mixed-traffic suburban expressway merges.",
        "path": [
            [77.2090, 28.6139],  # Delhi
            [76.9230, 28.3240],  # Gurgaon (Khedki Daula / Bilaspur Blackspot)
            [76.7941, 28.2054],  # Dharuhera
            [75.9597, 27.3917],  # Shahpura
            [75.7873, 26.9124],  # Jaipur
            [74.6399, 26.4499],  # Ajmer
            [73.7125, 24.5854],  # Udaipur
            [72.9698, 23.5977],  # Himatnagar
            [72.5714, 23.0225],  # Ahmedabad
            [73.1812, 22.3072],  # Vadodara
            [72.9959, 21.7051],  # Bharuch
            [72.8311, 21.1702],  # Surat
            [72.9106, 20.3893],  # Vapi
            [72.9781, 19.2183],  # Thane
            [72.8777, 19.0760],  # Mumbai
            [73.8180, 18.4550],  # Pune (Navale Bridge Fatal Descent Blackspot)
            [74.0183, 17.6805],  # Satara
            [74.2433, 16.7050],  # Kolhapur
            [74.4977, 15.8497],  # Belagavi
            [75.1240, 15.3647],  # Hubballi
            [75.9218, 14.4644],  # Davanagere
            [77.1010, 13.3409],  # Tumakuru
            [77.6100, 12.9160],  # Bengaluru (Silk Board & Electronic City Blackspot)
            [78.2138, 12.5186],  # Krishnagiri
            [79.1325, 12.9165],  # Vellore
            [79.7036, 12.8342],  # Kanchipuram
            [80.2707, 13.0827]   # Chennai
        ]
    },
    {
        "highway_id": "MUM-PUN-EXP",
        "highway_name": "Mumbai-Pune Expressway (Yashwantrao Chavan Expressway)",
        "short_name": "Mumbai-Pune Expressway",
        "length_km": 94.5,
        "category": "Expressway",
        "danger_level": "Extreme Danger",
        "color": [239, 68, 68, 240],       # Crimson Red
        "highlight_color": [255, 255, 255, 255],
        "total_blackspots": 52,
        "description": "India's first 6-lane access-controlled concrete toll expressway through the Western Ghats. High fatality rate driven by brake failure on steep Bhor Ghat slopes and lane-weaving at toll approaches.",
        "path": [
            [73.1070, 19.0183],  # Kalamboli Panvel Zero Point
            [73.2000, 18.9400],  # Shedung
            [73.3320, 18.8470],  # Khalapur Toll Plaza Blackspot
            [73.3660, 18.7880],  # Adoshi Tunnel Merge Blackspot
            [73.3750, 18.7610],  # Khandala Ghat Pass
            [73.4070, 18.7550],  # Lonavala Bhor Ghat Descent Blackspot
            [73.5550, 18.7580],  # Kamshet Tunnel
            [73.6800, 18.7300],  # Talegaon Toll
            [73.7380, 18.6650],  # Kiwale / Dehu Road Interchange
            [73.8000, 18.5800]   # Pune Bypass
        ]
    },
    {
        "highway_id": "YAM-EXP",
        "highway_name": "Yamuna Expressway (Greater Noida to Agra)",
        "short_name": "Yamuna Expressway",
        "length_km": 165.5,
        "category": "Expressway",
        "danger_level": "Critical Danger",
        "color": [244, 63, 94, 230],       # Rose / Hot Pink
        "highlight_color": [255, 255, 255, 255],
        "total_blackspots": 68,
        "description": "6-lane concrete access-controlled speed corridor through western UP. Infamous for highway hypnosis, high friction tire blowouts in summer, and catastrophic multi-vehicle pileups during winter radiation fog.",
        "path": [
            [77.5020, 28.4720],  # Greater Noida Pari Chowk Blackspot
            [77.5500, 28.3500],  # Dankaur Toll Stretch Blackspot
            [77.6200, 28.1800],  # Jewar Airport Interchange
            [77.6500, 27.9000],  # Bajna Toll
            [77.6800, 27.5500],  # Mathura - Vrindavan Toll Segment Blackspot
            [77.9500, 27.2400],  # Khandauli
            [78.1000, 27.2000]   # Kuberpur Agra Interchange
        ]
    },
    {
        "highway_id": "NH-66",
        "highway_name": "NH-66: Western Coastal Highway (Panvel - Goa - Mangaluru - Kochi - Kanyakumari)",
        "short_name": "NH-66",
        "length_km": 1608,
        "category": "National Highway",
        "danger_level": "Extreme Danger (Ghats & Blind Curves)",
        "color": [168, 85, 247, 220],     # Purple
        "highlight_color": [239, 68, 68, 255],
        "total_blackspots": 142,
        "description": "Scenic Konkan coastal arterial traversing rugged Western Ghats spurs. Extreme danger from undivided ghat sections, blind hairpin bends, steep valley drop-offs, and heavy monsoon landslides.",
        "path": [
            [73.1175, 18.9894],  # Panvel Mumbai
            [73.0934, 18.7352],  # Pen
            [73.4167, 18.0833],  # Mahad
            [73.4500, 17.8500],  # Kashedi Ghat & Chiplun Hairpin Curves Blackspot
            [73.5186, 17.5323],  # Chiplun
            [73.3120, 16.9902],  # Ratnagiri
            [73.7120, 16.2690],  # Kankavli
            [73.8278, 15.4909],  # Panaji / Goa
            [74.1299, 14.8159],  # Karwar
            [74.7421, 13.3409],  # Udupi
            [74.8560, 12.9141],  # Mangaluru
            [75.3704, 11.8745],  # Kannur
            [75.7804, 11.2588],  # Kozhikode
            [76.2673, 9.9312],   # Kochi
            [76.3388, 9.4981],   # Alappuzha
            [76.6141, 8.8932],   # Kollam
            [76.9366, 8.5241],   # Thiruvananthapuram
            [77.5385, 8.0883]    # Kanyakumari
        ]
    },
    {
        "highway_id": "SAMRUDDHI-EXP",
        "highway_name": "Samruddhi Mahamarg (Mumbai-Nagpur Super Communication Expressway)",
        "short_name": "Samruddhi Mahamarg",
        "length_km": 701,
        "category": "Expressway",
        "danger_level": "High Danger (Speed & Fatigue)",
        "color": [16, 185, 129, 220],      # Emerald Green
        "highlight_color": [239, 68, 68, 255],
        "total_blackspots": 44,
        "description": "Greenfield 120 km/h expressway linking Maharashtra's two capitals. Features wide open monotonous tangents leading to severe highway hypnosis and tire blowouts in extreme Vidarbha heat.",
        "path": [
            [79.0200, 21.0500],  # Nagpur Shivmadka Zero Point
            [78.6022, 20.7453],  # Wardha
            [77.7523, 20.9320],  # Amravati
            [77.4800, 20.4800],  # Karanja Lad
            [76.1800, 20.5200],  # Buldhana - Aurangabad Stretch Blackspot
            [75.8816, 19.8347],  # Jalna
            [75.3433, 19.8762],  # Aurangabad (Chhatrapati Sambhajinagar)
            [74.4766, 19.7667],  # Shirdi
            [73.7898, 19.9975],  # Sinnar Nashik
            [73.5600, 19.6974],  # Igatpuri Ghat
            [73.0631, 19.2967]   # Bhiwandi / Thane Interchange
        ]
    },
    {
        "highway_id": "NH-16",
        "highway_name": "NH-16: Golden Quadrilateral East Coast (Kolkata - Visakhapatnam - Chennai)",
        "short_name": "NH-16",
        "length_km": 1711,
        "category": "National Highway",
        "danger_level": "High Danger",
        "color": [59, 130, 246, 220],      # Blue
        "highlight_color": [239, 68, 68, 255],
        "total_blackspots": 126,
        "description": "Crucial Bay of Bengal commercial transit corridor. High collision densities near industrial ports where heavy container trucks encounter two-wheelers and unscientific pedestrian median crossings.",
        "path": [
            [88.3639, 22.5726],  # Kolkata
            [87.2320, 22.3460],  # Kharagpur
            [86.9135, 21.4934],  # Balasore
            [85.8828, 20.4625],  # Cuttack
            [85.8245, 20.2961],  # Bhubaneswar
            [84.7941, 19.3149],  # Brahmapur
            [83.8938, 18.2949],  # Srikakulam
            [83.2180, 17.6860],  # Visakhapatnam - Anakapalle Blackspot
            [81.8040, 17.0005],  # Rajahmundry
            [80.6480, 16.5062],  # Vijayawada
            [80.4365, 16.3067],  # Guntur
            [80.0499, 15.5057],  # Ongole
            [79.9865, 14.4426],  # Nellore
            [80.2707, 13.0827]   # Chennai
        ]
    },
    {
        "highway_id": "NH-19",
        "highway_name": "NH-19: Grand Trunk Road (Delhi - Kanpur - Varanasi - Kolkata)",
        "short_name": "NH-19 (GT Road)",
        "length_km": 1435,
        "category": "National Highway",
        "danger_level": "High Danger",
        "color": [234, 179, 8, 220],       # Gold / Yellow
        "highlight_color": [239, 68, 68, 255],
        "total_blackspots": 160,
        "description": "Historic trans-Gangetic trade spine of Northern and Eastern India. Heavy congestion, slow-moving agricultural tractors, overloaded multi-axle freight carriers, and dense winter fog.",
        "path": [
            [77.2090, 28.6139],  # Delhi
            [77.3178, 28.4089],  # Faridabad
            [77.6737, 27.4924],  # Mathura
            [78.0081, 27.1767],  # Agra
            [78.3957, 27.1593],  # Firozabad
            [79.0233, 26.7769],  # Etawah
            [80.3319, 26.4499],  # Kanpur Bypass Blackspot
            [81.8463, 25.4358],  # Prayagraj
            [82.9739, 25.3176],  # Varanasi Outer Blackspot
            [84.0150, 24.9520],  # Sasaram
            [86.4304, 23.7957],  # Dhanbad
            [86.9524, 23.6739],  # Asansol
            [87.3119, 23.5204],  # Durgapur
            [87.8615, 23.2324],  # Bardhaman
            [88.3639, 22.5726]   # Kolkata
        ]
    },
    {
        "highway_id": "HYD-ORR",
        "highway_name": "Outer Ring Road (ORR) Hyderabad Expressway",
        "short_name": "Hyderabad ORR",
        "length_km": 158,
        "category": "Expressway",
        "danger_level": "High Danger (Speeding)",
        "color": [139, 92, 246, 220],      # Violet
        "highlight_color": [239, 68, 68, 255],
        "total_blackspots": 38,
        "description": "8-lane ring expressway encircling Hyderabad with 100-120 km/h design speed. Major hazards include late-night high-speed car racing, sudden lane cuts at flyover exits, and collisions with stopped maintenance vehicles.",
        "path": [
            [78.3800, 17.3200],  # Shamshabad Airport Interchange Blackspot
            [78.5000, 17.2700],  # Tukkuguda
            [78.6000, 17.3400],  # Bongloor
            [78.6800, 17.4300],  # Ghatkesar
            [78.6000, 17.5800],  # Keesara
            [78.5000, 17.6100],  # Medchal
            [78.3600, 17.5600],  # Dundigal
            [78.2700, 17.4800],  # Patancheru
            [78.3400, 17.4300],  # Gachibowli Loop Blackspot
            [78.3700, 17.3800],  # Narsingi
            [78.3800, 17.3200]   # Shamshabad (Enclosed Loop)
        ]
    },
    {
        "highway_id": "NE-4-EXP",
        "highway_name": "Delhi-Mumbai Expressway (NE-4 Greenfield Corridor)",
        "short_name": "Delhi-Mumbai Expressway",
        "length_km": 1386,
        "category": "Expressway",
        "danger_level": "High Danger (120 km/h Wildlife & Hypnosis)",
        "color": [14, 165, 233, 220],      # Sky Blue
        "highlight_color": [239, 68, 68, 255],
        "total_blackspots": 42,
        "description": "State-of-the-art 8-lane expressway with speed limit 120 km/h. High-speed risks include driver drowsiness over immense uninterrupted straights and occasional stray cattle breaches.",
        "path": [
            [77.0600, 28.2500],  # Sohna (Delhi NCR Zero Point)
            [76.7500, 27.6500],  # Ferozepur Jhirka
            [76.3300, 26.8900],  # Dausa Rajasthan Blackspot
            [76.3500, 25.9900],  # Sawai Madhopur
            [75.8300, 25.1800],  # Kota Interchange
            [75.0400, 23.3300],  # Ratlam MP
            [73.1800, 22.3000],  # Vadodara
            [72.8300, 21.1700],  # Surat
            [72.9500, 18.9500]   # Mumbai JNPT Corridor
        ]
    }
]

# Grounded MoRTH Blackspots and Dangerous Highway Corridors in India
INDIA_HIGHWAY_CORRIDORS = [
    {
        "highway_id": "MUM-PUN-EXP",
        "highway_name": "Mumbai-Pune Expressway (Yashwantrao Chavan Expressway)",
        "state": "Maharashtra",
        "area_name": "Lonavala / Khandala Ghat Descent (Amrutanjan / Bhor Ghat)",
        "latitude": 18.7550,
        "longitude": 73.4070,
        "posted_speed": 100,
        "advisory_speed": 50,
        "danger_level": "Extreme Danger",
        "primary_cause": "Steep gradient descent, brake fade on heavy multi-axle trucks, and high-speed monsoon skidding.",
        "morth_blackspot_id": "MH-EXP-001",
        "fatalities_3yr": 74,
        "rectification_status": "Chicane rumble strips and escape ramps operational; mandatory gear downshifts enforced.",
        "weight": 0.16
    },
    {
        "highway_id": "MUM-PUN-EXP",
        "highway_name": "Mumbai-Pune Expressway (Yashwantrao Chavan Expressway)",
        "state": "Maharashtra",
        "area_name": "Khalapur Toll Plaza & Adoshi Tunnel Approach",
        "latitude": 18.8470,
        "longitude": 73.3320,
        "posted_speed": 100,
        "advisory_speed": 60,
        "danger_level": "Critical Danger",
        "primary_cause": "High-speed lane-weaving before toll queues and tunnel blind merges.",
        "morth_blackspot_id": "MH-EXP-002",
        "fatalities_3yr": 48,
        "rectification_status": "Speed-enforcement gantries installed; queue warning signboards deployed.",
        "weight": 0.10
    },
    {
        "highway_id": "YAM-EXP",
        "highway_name": "Yamuna Expressway (Greater Noida to Agra)",
        "state": "Uttar Pradesh",
        "area_name": "Zero Point Greater Noida / Dankaur Corridor",
        "latitude": 28.4720,
        "longitude": 77.5020,
        "posted_speed": 100,
        "advisory_speed": 75,
        "danger_level": "Extreme Danger",
        "primary_cause": "Highway hypnosis (driver drowsiness on monotonous concrete), tire bursts from friction heat, and dense winter fog pileups.",
        "morth_blackspot_id": "UP-EXP-014",
        "fatalities_3yr": 89,
        "rectification_status": "Nitrogen air-filling stations made free at tolls; speed limit dropped to 75 km/h during Dec-Jan fog.",
        "weight": 0.14
    },
    {
        "highway_id": "YAM-EXP",
        "highway_name": "Yamuna Expressway (Greater Noida to Agra)",
        "state": "Uttar Pradesh",
        "area_name": "Mathura - Vrindavan Milestone Toll Segment",
        "latitude": 27.5500,
        "longitude": 77.6800,
        "posted_speed": 100,
        "advisory_speed": 70,
        "danger_level": "Critical Danger",
        "primary_cause": "Excessive speeding exceeding 120 km/h and running into stationary disabled vehicles.",
        "morth_blackspot_id": "UP-EXP-022",
        "fatalities_3yr": 52,
        "rectification_status": "24x7 highway route patrol with dynamic VMS message boards.",
        "weight": 0.09
    },
    {
        "highway_id": "NH-48",
        "highway_name": "NH-48 (Delhi-Mumbai-Pune-Chennai)",
        "state": "Maharashtra",
        "area_name": "Navale Bridge & Katraj Dehu Road Bypass (Pune)",
        "latitude": 18.4550,
        "longitude": 73.8180,
        "posted_speed": 60,
        "advisory_speed": 40,
        "danger_level": "Extreme Danger",
        "primary_cause": "Continuous 4-km steep downhill slope into city traffic, brake failure on trucks, and stationary traffic pileups.",
        "morth_blackspot_id": "MH-NH48-038",
        "fatalities_3yr": 96,
        "rectification_status": "Speed limit lowered to 40 km/h, transverse rumble strips installed, truck brake inspection bays set up.",
        "weight": 0.13
    },
    {
        "highway_id": "NH-48",
        "highway_name": "NH-48 (Delhi-Jaipur Corridor)",
        "state": "Haryana",
        "area_name": "Khedki Daula / Bilaspur Chowk (Gurgaon / Manesar)",
        "latitude": 28.3240,
        "longitude": 76.9230,
        "posted_speed": 80,
        "advisory_speed": 50,
        "danger_level": "High Danger",
        "primary_cause": "Chaotic mixed traffic: two-wheelers and pedestrians crossing high-speed express lanes.",
        "morth_blackspot_id": "HR-NH48-011",
        "fatalities_3yr": 61,
        "rectification_status": "Pedestrian foot overbridge constructed; service road barriers deployed.",
        "weight": 0.11
    },
    {
        "highway_id": "NH-44",
        "highway_name": "NH-44 (Kashmir to Kanyakumari)",
        "state": "Punjab",
        "area_name": "Sherpur Chowk & Basti Jodhewal (Ludhiana)",
        "latitude": 30.8980,
        "longitude": 75.8920,
        "posted_speed": 70,
        "advisory_speed": 45,
        "danger_level": "High Danger",
        "primary_cause": "Urban bottleneck congestion, high-volume commercial transit, and lack of segregated service lanes.",
        "morth_blackspot_id": "PB-NH44-007",
        "fatalities_3yr": 54,
        "rectification_status": "Elevated corridor constructed; grade-separation at major junctions completed.",
        "weight": 0.08
    },
    {
        "highway_id": "NH-44",
        "highway_name": "NH-44 (Delhi-Haryana Corridor)",
        "state": "Haryana",
        "area_name": "Murthal - Panipat Toll Corridor (Kundli Border)",
        "latitude": 28.9930,
        "longitude": 77.0150,
        "posted_speed": 90,
        "advisory_speed": 60,
        "danger_level": "Critical Danger",
        "primary_cause": "Late-night high-speed drunk driving, illegal median cuts, and winter radiation fog.",
        "morth_blackspot_id": "HR-NH44-019",
        "fatalities_3yr": 78,
        "rectification_status": "Median cuts closed with steel crash barriers; automated ANPR speed cameras installed.",
        "weight": 0.09
    },
    {
        "highway_id": "NH-44",
        "highway_name": "NH-44 (Hyderabad-Bengaluru Corridor)",
        "state": "Andhra Pradesh",
        "area_name": "Kurnool - Anantapur Highway Stretch",
        "latitude": 15.8280,
        "longitude": 78.0370,
        "posted_speed": 90,
        "advisory_speed": 65,
        "danger_level": "High Danger",
        "primary_cause": "High-speed long-distance night driving fatigue and uncontrolled cattle/tractor crossings.",
        "morth_blackspot_id": "AP-NH44-029",
        "fatalities_3yr": 43,
        "rectification_status": "Retroreflective cat-eyes installed, cattle underpasses built at critical agricultural points.",
        "weight": 0.07
    },
    {
        "highway_id": "NH-48",
        "highway_name": "NH-48 (Bengaluru-Hosur Highway)",
        "state": "Karnataka",
        "area_name": "Silk Board & Electronic City Elevated Toll Descent (Bengaluru)",
        "latitude": 12.9160,
        "longitude": 77.6100,
        "posted_speed": 80,
        "advisory_speed": 40,
        "danger_level": "High Danger",
        "primary_cause": "Abrupt speed reduction off elevated flyovers, heavy two-wheeler weaving, and sudden braking.",
        "morth_blackspot_id": "KA-NH48-004",
        "fatalities_3yr": 59,
        "rectification_status": "Speed deceleration zones marked with thermo-plastic rumble stripes.",
        "weight": 0.10
    },
    {
        "highway_id": "HYD-ORR",
        "highway_name": "Outer Ring Road (ORR) Hyderabad",
        "state": "Telangana",
        "area_name": "Shamshabad Airport Interchange & Gachibowli Loop",
        "latitude": 17.3200,
        "longitude": 78.3800,
        "posted_speed": 100,
        "advisory_speed": 70,
        "danger_level": "High Danger",
        "primary_cause": "High-speed racing, midnight tire blowouts, and crashes into stationary maintenance trucks.",
        "morth_blackspot_id": "TG-ORR-002",
        "fatalities_3yr": 46,
        "rectification_status": "Strict speed laser interceptors and LED illuminated lane delineators operational.",
        "weight": 0.08
    },
    {
        "highway_id": "NH-66",
        "highway_name": "NH-66 (Mumbai-Goa Highway)",
        "state": "Maharashtra",
        "area_name": "Kashedi Ghat & Chiplun Curve Pass",
        "latitude": 17.8500,
        "longitude": 73.4500,
        "posted_speed": 60,
        "advisory_speed": 35,
        "danger_level": "Extreme Danger",
        "primary_cause": "Undivided sharp blind hairpin curves, steep valley drops, and heavy monsoon landslides.",
        "morth_blackspot_id": "MH-NH66-021",
        "fatalities_3yr": 68,
        "rectification_status": "Twin-tube Kashedi Ghat Tunnel opened to bypass treacherous hairpin curves.",
        "weight": 0.08
    },
    {
        "highway_id": "SAMRUDDHI-EXP",
        "highway_name": "Samruddhi Mahamarg (Mumbai-Nagpur Expressway)",
        "state": "Maharashtra",
        "area_name": "Buldhana - Aurangabad (Chhatrapati Sambhajinagar) Stretch",
        "latitude": 20.5200,
        "longitude": 76.1800,
        "posted_speed": 120,
        "advisory_speed": 80,
        "danger_level": "Critical Danger",
        "primary_cause": "Highway hypnosis, high summer heat causing explosive tire bursts, and excessive speeding.",
        "morth_blackspot_id": "MH-EXP-088",
        "fatalities_3yr": 82,
        "rectification_status": "Driver psychological rest facilities with audio rumble warnings installed every 50 km.",
        "weight": 0.09
    },
    {
        "highway_id": "NH-16",
        "highway_name": "NH-16 (Kolkata-Chennai Coastal Corridor)",
        "state": "Andhra Pradesh",
        "area_name": "Visakhapatnam - Anakapalle Industrial Stretch",
        "latitude": 17.6860,
        "longitude": 83.2180,
        "posted_speed": 80,
        "advisory_speed": 50,
        "danger_level": "High Danger",
        "primary_cause": "Heavy multi-axle freight movement, unscientific U-turns, and night pedestrian crossing.",
        "morth_blackspot_id": "AP-NH16-015",
        "fatalities_3yr": 55,
        "rectification_status": "Service road segregation and 6 flyovers under accelerated execution.",
        "weight": 0.07
    }
]


def get_india_highway_routes() -> List[Dict[str, Any]]:
    """Return all India highway network polylines with danger attributes for PyDeck PathLayer."""
    return INDIA_HIGHWAY_NETWORKS


def get_highway_corridors_list() -> List[Dict[str, Any]]:
    """Return the list of MoRTH blackspot corridors with danger levels."""
    return INDIA_HIGHWAY_CORRIDORS


def generate_india_accidents_dataset(
    output_path: str = "data/raw/india_accident_blackspots.csv",
    n_samples: int = 2500,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate calibrated accident records across India's premier blackspots
    grounded in MoRTH (Ministry of Road Transport and Highways) annual accident metrics.
    """
    logger.info(f"Generating India Highway Accident Blackspots Dataset ({n_samples} records, seed={random_state})...")
    rng = np.random.default_rng(random_state)

    corridor_weights = np.array([c["weight"] for c in INDIA_HIGHWAY_CORRIDORS])
    corridor_weights /= corridor_weights.sum()

    chosen_indices = rng.choice(len(INDIA_HIGHWAY_CORRIDORS), size=n_samples, p=corridor_weights)

    data_rows = []
    for i, idx in enumerate(chosen_indices):
        c = INDIA_HIGHWAY_CORRIDORS[idx]

        # Small spatial dispersion along the corridor road stretch (~0.012 to 0.020 degrees)
        lat = np.round(c["latitude"] + rng.normal(0, 0.015), 5)
        lon = np.round(c["longitude"] + rng.normal(0, 0.015), 5)

        # Vehicle type distribution typical for Indian highways
        veh = rng.choice(["Car", "Motorcycle", "Goods vehicle (Truck/Lorry)", "Bus", "Auto Rickshaw"],
                         p=[0.42, 0.28, 0.18, 0.08, 0.04])

        # Weather & lighting
        weather = rng.choice(["Clear", "Dense Winter Fog", "Heavy Monsoon Rain", "Dust Storm / Haze"],
                             p=[0.70, 0.12, 0.14, 0.04])
        light = rng.choice(["Daylight", "Night - Streetlights Lit", "Night - Pitch Dark / Unlit", "Dawn/Dusk"],
                           p=[0.58, 0.20, 0.16, 0.06])

        # Actual recorded speed
        posted = c["posted_speed"]
        recorded_speed = int(np.clip(rng.normal(posted + 8, 15), 30, 160))

        # Severity probability based on corridor hazards & speeding
        risk_score = 0.0
        if recorded_speed > c["advisory_speed"] + 25:
            risk_score += 1.4
        elif recorded_speed > c["advisory_speed"] + 10:
            risk_score += 0.8

        if veh in ["Motorcycle", "Auto Rickshaw"]:
            risk_score += 1.5
        elif veh == "Goods vehicle (Truck/Lorry)":
            risk_score += 0.6

        if light == "Night - Pitch Dark / Unlit":
            risk_score += 1.1

        if weather in ["Dense Winter Fog", "Heavy Monsoon Rain"]:
            risk_score += 0.8

        # In Indian MoRTH blackspot zones, severity is notoriously high
        # Base: Fatal ~9%, Serious ~36%, Slight ~55%
        p_fatal = np.clip(0.09 + (risk_score * 0.05), 0.03, 0.35)
        p_serious = np.clip(0.36 + (risk_score * 0.04), 0.15, 0.55)
        p_slight = 1.0 - (p_fatal + p_serious)
        p_slight = max(0.10, p_slight)

        tot_p = p_slight + p_serious + p_fatal
        p_slight /= tot_p
        p_serious /= tot_p
        p_fatal /= tot_p

        severity = rng.choice(["Slight", "Serious", "Fatal"], p=[p_slight, p_serious, p_fatal])

        data_rows.append({
            "accident_id": f"IND-2026-{i+1:05d}",
            "highway_id": c.get("highway_id", "NH-48"),
            "highway_name": c["highway_name"],
            "state": c["state"],
            "area_name": c["area_name"],
            "latitude": lat,
            "longitude": lon,
            "posted_speed_kmh": posted,
            "advisory_safe_speed_kmh": c["advisory_speed"],
            "danger_level": c.get("danger_level", "High Danger"),
            "recorded_speed_kmh": recorded_speed,
            "vehicle_type": veh,
            "weather_conditions": weather,
            "light_conditions": light,
            "accident_severity": severity,
            "primary_cause": c["primary_cause"],
            "morth_blackspot_id": c["morth_blackspot_id"],
            "rectification_status": c.get("rectification_status", "Active monitoring")
        })

    df = pd.DataFrame(data_rows)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved India Highway Accidents Dataset to: {output_path}")

    return df


def load_india_accidents_dataset(file_path: str = "data/raw/india_accident_blackspots.csv") -> pd.DataFrame:
    """Load India accident blackspots dataset, generating if absent."""
    if not os.path.exists(file_path):
        return generate_india_accidents_dataset(file_path)
    df = pd.read_csv(file_path)
    # If highway_id or danger_level missing from earlier run, re-generate to keep schema synchronized
    if "highway_id" not in df.columns or "danger_level" not in df.columns:
        return generate_india_accidents_dataset(file_path)
    return df
