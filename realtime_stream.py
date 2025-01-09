import os
import pyshark
import psutil
from collections import defaultdict

# Define the path to tshark executable (modify this according to your system)
tshark_path = 'D:/Wireshark/tshark.exe'

# Get a list of available network interfaces
def get_interfaces():
    addrs = psutil.net_if_addrs()
    return list(addrs.keys())

def preprocess_packet(packet):
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
        icmp_type = int(packet['ICMP'].get_field_value('type')) if 'ICMP' in packet else None
        icmp_code = int(packet['ICMP'].get_field_value('code')) if 'ICMP' in packet else None
        dns_query = packet['DNS'].get_field_value('qname') if 'DNS' in packet else None

        # Extract timestamp from the packet
        timestamp = float(packet.sniff_timestamp)

        # Display the preprocessed packet information
        print(f"Timestamp: {timestamp}")
        print(f"Source IP: {source_ip}")
        print(f"Destination IP: {destination_ip}")
        print(f"Protocol: {protocol}")
        print(f"Total Length: {total_length}")
        print(f"TTL: {ttl}")
        print(f"Flags: {flags}")
        print(f"Source Port: {source_port}")
        print(f"Destination Port: {destination_port}")
        print(f"TCP Flags: {tcp_flags}")
        print(f"ICMP Type: {icmp_type}")
        print(f"ICMP Code: {icmp_code}")
        print(f"DNS Query: {dns_query}")
        print("\n")

# List all available network interfaces
network_interfaces = get_interfaces()

print("Available network interfaces:")
for idx, interface in enumerate(network_interfaces, start=1):
    print(f"{idx}. {interface}")

# Ask the user to select a network interface
selected_idx = int(input("Select a network interface (enter the corresponding number): "))
selected_interface = network_interfaces[selected_idx - 1]
print(f"Monitoring network interface: {selected_interface}")

# Capture packets on the selected interface
capture = pyshark.LiveCapture(interface=selected_interface, tshark_path=tshark_path)

# Continuously process and display packets
for packet in capture.sniff_continuously():
    preprocess_packet(packet)
