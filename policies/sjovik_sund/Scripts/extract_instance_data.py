#!/usr/bin/env python3
"""
Quick script to extract OS_W34 and other instance data to readable JSON files.
Saves to policies/sjovik_sund/output/ directory for easy inspection.
"""

import gzip
import json
import os
import sys
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
INSTANCES_DIR = PROJECT_ROOT / 'instances'
OUTPUT_DIR = SCRIPT_DIR.parent / 'output' / 'instance_data'


def extract_instance_to_json(instance_name):
    """
    Extract instance from .json.gz to readable .json file.
    
    Args:
        instance_name: Name like 'OS_W34', 'TD_W34', etc.
    """
    input_path = INSTANCES_DIR / f"{instance_name}.json.gz"
    
    if not input_path.exists():
        print(f"Error: {input_path} not found!")
        return False
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{instance_name}.json"
    
    print(f"\nExtracting {instance_name}...")
    print(f"  From: {input_path}")
    print(f"  To:   {output_path}")
    
    try:
        # Read compressed data
        with gzip.open(input_path, 'r') as f:
            data = json.load(f)
        
        # Write readable JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Print statistics
        print(f"\n[SUCCESS] Extracted {instance_name}")
        print(f"  Data keys: {list(data.keys())}")
        
        if 'stations' in data:
            stations = data['stations']
            print(f"  Stations: {len(stations)}")
            # Show first few station IDs (handle both dict and list format)
            if isinstance(stations, dict):
                station_ids = list(stations.keys())[:5]
                print(f"  First stations: {station_ids}")
            elif isinstance(stations, list) and len(stations) > 0:
                # Extract IDs from list of station objects
                station_ids = [s.get('id', s.get('name', i)) for i, s in enumerate(stations[:5])]
                print(f"  First stations: {station_ids}")
        
        if 'bikes' in data:
            print(f"  Bikes: {len(data['bikes'])}")
        
        if 'vehicles' in data:
            print(f"  Vehicles: {len(data['vehicles'])}")
        
        # File sizes
        size_gz = input_path.stat().st_size / 1024
        size_json = output_path.stat().st_size / 1024
        print(f"  Compressed: {size_gz:.1f} KB -> Uncompressed: {size_json:.1f} KB")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_instances(instance_list):
    """
    Extract multiple instances and create a comparison summary.
    """
    print("="*70)
    print("Instance Data Extraction & Comparison")
    print("="*70)
    
    results = {}
    
    for instance_name in instance_list:
        if extract_instance_to_json(instance_name):
            # Load the extracted data for comparison
            output_path = OUTPUT_DIR / f"{instance_name}.json"
            with open(output_path, 'r') as f:
                results[instance_name] = json.load(f)
    
    # Create comparison summary
    if len(results) > 1:
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        
        comparison_data = []
        for name, data in results.items():
            # Handle both dict and list formats
            stations = data.get('stations', [])
            bikes = data.get('bikes', [])
            vehicles = data.get('vehicles', [])
            
            comparison_data.append({
                'Instance': name,
                'Stations': len(stations) if stations else 0,
                'Bikes': len(bikes) if bikes else 0,
                'Vehicles': len(vehicles) if vehicles else 0,
                'Has_Map': 'map' in data,
            })
        
        # Print comparison table
        print(f"\n{'Instance':<15} {'Stations':>10} {'Bikes':>10} {'Vehicles':>10} {'Has Map':>10}")
        print("-"*70)
        for row in comparison_data:
            print(f"{row['Instance']:<15} {row['Stations']:>10} {row['Bikes']:>10} "
                  f"{row['Vehicles']:>10} {str(row['Has_Map']):>10}")
        
        # Save comparison to CSV
        comparison_path = OUTPUT_DIR / 'instances_comparison.csv'
        import csv
        with open(comparison_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=comparison_data[0].keys())
            writer.writeheader()
            writer.writerows(comparison_data)
        print(f"\n[SUCCESS] Comparison saved to: {comparison_path}")
    
    print(f"\n[SUCCESS] All files saved to: {OUTPUT_DIR}")
    print("="*70)


def main():
    """Main function."""
    if len(sys.argv) > 1:
        # Extract specific instances from command line
        instances = sys.argv[1:]
        compare_instances(instances)
    else:
        # Default: extract OS_W34, OS_W31, and TD_W34 for comparison
        print("No instances specified. Extracting default set...")
        print("Usage: python extract_instance_data.py <instance1> <instance2> ...")
        print("Example: python extract_instance_data.py OS_W34 OS_W31 TD_W34\n")
        
        default_instances = ['OS_W34', 'OS_W31', 'TD_W34']
        compare_instances(default_instances)


if __name__ == "__main__":
    main()
