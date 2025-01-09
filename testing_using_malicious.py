import time
import random
import os
import pyshark
import psutil
import pickle
import numpy as np
from collections import defaultdict
from statistics import mean, stdev
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model
from scapy.all import IP, ICMP, send ,TCP

# Define paths to pre-trained model and preprocessing objects
model_file = 'D:\Python files\Mini Project\model\deeper_hybrid_model.keras'
scaler_file = 'D:\Python files\Mini Project\model\scaler.pkl'
label_encoder_file = 'D:\Python files\Mini Project\model\label_encoder.pkl'

# Define the path to tshark executable (modify this according to your system)
tshark_path = 'D:/Wireshark/tshark.exe'

# Get a list of available network interfaces
def get_interfaces():
    addrs = psutil.net_if_addrs()
    return list(addrs.keys())

def preprocess_packet(packet, preprocessed_packets):
    if 'IP' in packet:
        ip_layer = packet['IP']
        source_ip = ip_layer.src
        destination_ip = ip_layer.dst
        protocol = ip_layer.get_field_value('proto')
        total_length = int(ip_layer.get_field_value('len'))
        ttl = int(ip_layer.get_field_value('ttl'))
        flags = ip_layer.get_field_value('flags')
        source_port = int(packet['TCP'].get_field_value('srcport')) if 'TCP' in packet else None
        destination_port = int(packet['TCP'].get_field_value('dstport')) if 'TCP' in packet else None
        tcp_flags = packet['TCP'].get_field_value('flags') if 'TCP' in packet else None

        # Extract timestamp from the packet
        timestamp = float(packet.sniff_timestamp)

        # Create a flow identifier by combining source and destination IPs and ports
        flow_id = (source_ip, destination_ip, source_port, destination_port)

        # Store the preprocessed packet information in a dictionary by flow ID
        preprocessed_packets[flow_id].append({
            'Timestamp': timestamp,
            'Source IP': source_ip,
            'Destination IP': destination_ip,
            'Protocol': protocol,
            'Total Length': total_length,
            'TTL': ttl,
            'Flags': flags,
            'Source Port': source_port,
            'Destination Port': destination_port,
            'TCP Flags': tcp_flags,
            # Add more attributes as required
        })

def calculate_flow_duration(packets):
    first_packet_time = packets[0]['Timestamp']
    last_packet_time = packets[-1]['Timestamp']
    return last_packet_time - first_packet_time

def calculate_bwd_packet_length_statistics(packets, target_ip):
    bwd_packet_lengths = [packet['Total Length'] for packet in packets if packet['Destination IP'] == target_ip]
    max_bwd_packet_length = max(bwd_packet_lengths)
    mean_bwd_packet_length = mean(bwd_packet_lengths)
    std_bwd_packet_length = stdev(bwd_packet_lengths)
    return max_bwd_packet_length, mean_bwd_packet_length, std_bwd_packet_length

def calculate_flow_iat_statistics(packets):
    flow_iats = [packets[i]['Timestamp'] - packets[i - 1]['Timestamp'] for i in range(1, len(packets))]
    flow_iat_std = stdev(flow_iats) if len(flow_iats) > 1 else 0
    flow_iat_max = max(flow_iats) if len(flow_iats) > 0 else 0
    return flow_iat_std, flow_iat_max

def calculate_fwd_iat_statistics(packets):
    fwd_iats = [packets[i]['Timestamp'] - packets[i - 1]['Timestamp'] for i in range(1, len(packets))]
    fwd_iat_std = stdev(fwd_iats) if len(fwd_iats) > 1 else 0
    fwd_iat_max = max(fwd_iats) if len(fwd_iats) > 0 else 0
    return fwd_iat_std, fwd_iat_max

def calculate_flow_bytes_per_second(packets, flow_duration):
    total_bytes = sum(packet['Total Length'] for packet in packets)
    return total_bytes / flow_duration

def calculate_fwd_iat_total(packets):
    fwd_iat_total = sum(packets[i]['Timestamp'] - packets[i - 1]['Timestamp'] for i in range(1, len(packets)))
    return fwd_iat_total

# Calculate other features based on the properties of the packets
def calculate_features(preprocessed_packets):
    features = []

    idle_intervals = defaultdict(list)  # Dictionary to store idle intervals for each flow

    for flow_id, flow_packets in preprocessed_packets.items():
        for i in range(1, len(flow_packets)):
            current_packet = flow_packets[i]
            previous_packet = flow_packets[i - 1]

            # Calculate time interval between packets
            time_interval = current_packet['Timestamp'] - previous_packet['Timestamp']
            idle_intervals[flow_id].append(time_interval)  # Store the interval for idle calculations

            # Calculate the total length of packets in forward and backward directions
            total_fwd_length = current_packet['Total Length']
            total_bwd_length = previous_packet['Total Length']

            # Calculate flow features
            flow_duration = calculate_flow_duration(flow_packets[i - 1:i + 1])

            max_bwd_packet_length, mean_bwd_packet_length, std_bwd_packet_length = calculate_bwd_packet_length_statistics(
                flow_packets[i - 1:i + 1], current_packet['Destination IP'])

            # Calculate flow features
            if time_interval == 0:
                flow_bytes_per_s = 0
            else:
                flow_bytes_per_s = (total_fwd_length + total_bwd_length) / time_interval

            flow_iat_std, flow_iat_max = calculate_flow_iat_statistics(flow_packets[i - 1:i + 1])

            fwd_iat_std, fwd_iat_max = calculate_fwd_iat_statistics(flow_packets[i - 1:i + 1])

            fwd_iat_total = calculate_fwd_iat_total(flow_packets[i - 1:i + 1]) 

            # Calculate packet length statistics
            packet_lengths = [packet['Total Length'] for packet in flow_packets[i - 1:i + 1]]
            mean_packet_length = mean(packet_lengths)

            # Calculate packet length standard deviation
            packet_length_std = (sum((x - mean_packet_length) ** 2 for x in packet_lengths) / (
                        len(packet_lengths) - 1)) ** 0.5

            # Calculate packet length variance
            packet_length_variance = packet_length_std ** 2

            max_packet_length = max(current_packet['Total Length'], previous_packet['Total Length'])

            packet_length_mean = (current_packet['Total Length'] + previous_packet['Total Length']) / 2

            psh_flag_count = 2 if current_packet['TCP Flags'] == 'PA' or previous_packet['TCP Flags'] == 'PA' else 0

            average_packet_size = (total_fwd_length + total_bwd_length) / 2

            avg_bwd_segment_size = total_bwd_length / i

            # Calculate idle features
            idle_mean = mean(idle_intervals[flow_id])
            idle_max = max(idle_intervals[flow_id])
            idle_min = min(idle_intervals[flow_id])

            feature_vector = {
                'Flow Duration': flow_duration,
                'Bwd Packet Length Max': max_bwd_packet_length,
                'Bwd Packet Length Mean': mean_bwd_packet_length,
                'Bwd Packet Length Std': std_bwd_packet_length,
                'Flow Bytes/s': flow_bytes_per_s,
                'Flow IAT Std': flow_iat_std,
                'Flow IAT Max': flow_iat_max,
                'Fwd IAT Std': fwd_iat_std,
                'Fwd IAT Max': fwd_iat_max,
                'Max Packet Length': max_packet_length,
                'Packet Length Mean': packet_length_mean,
                'Packet Length Std': packet_length_std,
                'Packet Length Variance': packet_length_variance,
                'PSH Flag Count': psh_flag_count,
                'Average Packet Size': average_packet_size,
                'Avg Bwd Segment Size': avg_bwd_segment_size,
                'Idle Mean': idle_mean,
                'Idle Max': idle_max,
                'Idle Min': idle_min,
                'Fwd IAT Total': fwd_iat_total,
            }

            features.append(feature_vector)

    return features

def extract_single_feature_vector(features):
    # Calculate the average feature vector
    num_features = len(features)
    if num_features == 0:
        return None  # Return None if there are no features
    else:
        average_feature_vector = {}
        for feature_name in features[0].keys():
            feature_values = [feature[feature_name] for feature in features]
            average_value = mean(feature_values)
            average_feature_vector[feature_name] = average_value
        return average_feature_vector

# Function to load pre-trained model and preprocessing objects
def load_pretrained_model(model_file, scaler_file, label_encoder_file):
    model = load_model(model_file)
    with open(scaler_file, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    with open(label_encoder_file, 'rb') as label_encoder_file:
        label_encoder = pickle.load(label_encoder_file)
    return model, scaler, label_encoder

class MaliciousTrafficGenerator:
    def __init__(self):
        # Use a broadcast address as the destination IP for ICMP packets
        self.target_ip = "192.168.0.15"

    def send_malicious_traffic(self, packet_count=500):
        for i in range(packet_count):
                    # Craft a malicious packet with anomalous values
                    malicious_packet = IP(dst=self.target_ip) / TCP(flags="A", window=10000)  # Adjust values as needed

                    # Send the malicious packet
                    send(malicious_packet)

                    # Print a message indicating that a malicious packet was sent
                    print(f"Malicious packet {i + 1} sent: {malicious_packet.summary()}")

class RedTeamTester:
    def __init__(self, malicious_traffic_generator, ids_model, selected_interface):
        self.malicious_traffic_generator = malicious_traffic_generator
        self.ids_model = ids_model
        self.selected_interface = selected_interface
        self.malignant_count = 0  # Initialize the count of malignant detections
        
    def run_tests(self, test_duration=500):
        start_time = time.time()  # Record the start time
        end_time = start_time + test_duration  # Calculate the end time

        while time.time() < end_time and self.malignant_count == 0:
            # Generate and send malicious traffic
            self.malicious_traffic_generator.send_malicious_traffic()

            # Modify the filter to capture all traffic on the selected interface
            capture = pyshark.LiveCapture(interface=self.selected_interface, tshark_path=tshark_path, display_filter="")
      

            # Initialize variables for preprocessing
            preprocessed_packets = defaultdict(list)

            # Continuously process and display packets
            for packet in capture.sniff_continuously():
                preprocess_packet(packet, preprocessed_packets)

                # Calculate features based on preprocessed packets
                features = calculate_features(preprocessed_packets)

                if len(features) > 0:
                    feature_vector = extract_single_feature_vector(features)
                    if feature_vector is not None:
                        feature_array = np.array([list(feature_vector.values())])
                        scaled_feature_vector = scaler.transform(feature_array)

                        # Make a prediction using the pre-trained model
                        prediction = self.ids_model.predict(scaled_feature_vector)

                        # Reshape the prediction array to 1D
                        prediction = prediction.ravel()

                        # Use the label encoder to decode the prediction
                        decoded_prediction = label_encoder.inverse_transform([np.argmax(prediction)])

                        # Display whether the packet is benign or malignant
                        if decoded_prediction[0] == "BENIGN":
                            print("Packet Prediction: BENIGN")
                        else:
                            print("Packet Prediction: MALIGNANT")
                            self.malignant_count += 1  # Increment the malignant count

                # Check if the test duration has been exceeded and exit the loop if necessary
                current_time = time.time() - start_time
                if current_time >= test_duration:
                    break

            # Sleep for a short duration to avoid continuous processing
            time.sleep(0.1)  # Adjust the sleep duration as needed

        # After the tests are completed, print the number of malignant packets detected
        print(f"Number of Malignant Detections: {self.malignant_count}")


if __name__ == "__main__":
    # Create an instance of the malicious traffic generator
    malicious_traffic_generator = MaliciousTrafficGenerator()

    # Initialize the pre-trained model, scaler, and label encoder
    model, scaler, label_encoder = load_pretrained_model(model_file, scaler_file, label_encoder_file)

    # List all available network interfaces
    network_interfaces = get_interfaces()

    print("Available network interfaces:")
    for idx, interface in enumerate(network_interfaces, start=1):
        print(f"{idx}. {interface}")

    # Ask the user to select a network interface
    selected_idx = int(input("Select a network interface (enter the corresponding number): "))
    selected_interface = network_interfaces[selected_idx - 1]
    print(f"Monitoring network interface: {selected_interface}")

    # Create an instance of the RedTeamTester class
    red_team_tester = RedTeamTester(malicious_traffic_generator, model, selected_interface)

    # Run red team tests for exactly 2 minutes and display the number of malignant detections
    red_team_tester.run_tests(test_duration=60)
