#!/usr/bin/env python3
"""
Analysis and visualization script for bee detection results.

This script analyzes detection results from BeeDetector and creates visualizations
and summary statistics. It can be run independently or imported as a module.
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_detection_data(csv_path):
    """
    Load detection data from CSV file.
    
    Args:
        csv_path (str): Path to the detection CSV file
        
    Returns:
        pd.DataFrame: Detection data
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} detections from {csv_path}")
        return df
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)


def create_visualizations(df, output_dir):
    """
    Create visualization plots for detection analysis.
    
    Args:
        df (pd.DataFrame): Detection data
        output_dir (str): Directory to save plots
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating visualizations in: {output_path}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Bee Detection Analysis', fontsize=16, fontweight='bold')
    
    # 1. Confidence distribution
    axes[0, 0].hist(df['confidence'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Confidence Score')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Detection Confidence Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add statistics text
    mean_conf = df['confidence'].mean()
    std_conf = df['confidence'].std()
    axes[0, 0].axvline(mean_conf, color='red', linestyle='--', 
                      label=f'Mean: {mean_conf:.3f}')
    axes[0, 0].legend()
    
    # 2. Detections per frame
    detections_per_frame = df.groupby('frame').size()
    axes[0, 1].plot(detections_per_frame.index, detections_per_frame.values, 
                   marker='o', markersize=3, linewidth=1)
    axes[0, 1].set_xlabel('Frame Number')
    axes[0, 1].set_ylabel('Number of Detections')
    axes[0, 1].set_title('Detections per Frame')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Track duration distribution
    track_duration = df.groupby('track_id')['frame'].agg(['min', 'max'])
    track_duration['duration'] = track_duration['max'] - track_duration['min'] + 1
    axes[1, 0].hist(track_duration['duration'], bins=20, alpha=0.7, 
                   color='lightgreen', edgecolor='black')
    axes[1, 0].set_xlabel('Track Duration (frames)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Track Duration Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add mean duration line
    mean_duration = track_duration['duration'].mean()
    axes[1, 0].axvline(mean_duration, color='red', linestyle='--',
                      label=f'Mean: {mean_duration:.1f} frames')
    axes[1, 0].legend()
    
    # 4. Confidence vs Track Duration scatter plot
    track_stats = df.groupby('track_id').agg({
        'confidence': 'mean',
        'frame': ['min', 'max']
    }).reset_index()
    track_stats.columns = ['track_id', 'avg_confidence', 'start_frame', 'end_frame']
    track_stats['duration'] = track_stats['end_frame'] - track_stats['start_frame'] + 1
    
    scatter = axes[1, 1].scatter(track_stats['duration'], track_stats['avg_confidence'], 
                               alpha=0.6, c=track_stats['track_id'], cmap='viridis')
    axes[1, 1].set_xlabel('Track Duration (frames)')
    axes[1, 1].set_ylabel('Average Confidence')
    axes[1, 1].set_title('Track Duration vs Average Confidence')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('Track ID')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = output_path / 'detection_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization plot: {plot_path}")
    
    # Also save as PDF for high quality
    pdf_path = output_path / 'detection_analysis.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved visualization plot (PDF): {pdf_path}")
    
    plt.show()


def generate_summary_statistics(df, output_dir):
    """
    Generate summary statistics and save to CSV.
    
    Args:
        df (pd.DataFrame): Detection data
        output_dir (str): Directory to save summary CSV
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate summary statistics
    summary_stats = []
    
    # Overall statistics
    summary_stats.append({
        'Metric': 'Total Detections',
        'Value': len(df),
        'Description': 'Total number of bee detections across all frames'
    })
    
    summary_stats.append({
        'Metric': 'Unique Tracks',
        'Value': df['track_id'].nunique(),
        'Description': 'Number of unique bee tracks identified'
    })
    
    summary_stats.append({
        'Metric': 'Frames with Detections',
        'Value': df['frame'].nunique(),
        'Description': 'Number of frames containing at least one detection'
    })
    
    # Confidence statistics
    summary_stats.append({
        'Metric': 'Average Confidence',
        'Value': round(df['confidence'].mean(), 4),
        'Description': 'Mean confidence score across all detections'
    })
    
    summary_stats.append({
        'Metric': 'Min Confidence',
        'Value': round(df['confidence'].min(), 4),
        'Description': 'Lowest confidence score'
    })
    
    summary_stats.append({
        'Metric': 'Max Confidence',
        'Value': round(df['confidence'].max(), 4),
        'Description': 'Highest confidence score'
    })
    
    summary_stats.append({
        'Metric': 'Confidence Std Dev',
        'Value': round(df['confidence'].std(), 4),
        'Description': 'Standard deviation of confidence scores'
    })
    
    # Track duration statistics
    track_duration = df.groupby('track_id')['frame'].agg(['min', 'max'])
    track_duration['duration'] = track_duration['max'] - track_duration['min'] + 1
    
    summary_stats.append({
        'Metric': 'Average Track Duration',
        'Value': round(track_duration['duration'].mean(), 1),
        'Description': 'Mean track duration in frames'
    })
    
    summary_stats.append({
        'Metric': 'Min Track Duration',
        'Value': int(track_duration['duration'].min()),
        'Description': 'Shortest track duration in frames'
    })
    
    summary_stats.append({
        'Metric': 'Max Track Duration',
        'Value': int(track_duration['duration'].max()),
        'Description': 'Longest track duration in frames'
    })
    
    # Detection density
    total_frames = df['frame'].max() - df['frame'].min() + 1 if len(df) > 0 else 0
    detection_density = len(df) / total_frames if total_frames > 0 else 0
    
    summary_stats.append({
        'Metric': 'Detection Density',
        'Value': round(detection_density, 4),
        'Description': 'Average detections per frame'
    })
    
    # Low confidence detections
    low_conf_threshold = 0.5
    low_conf_count = len(df[df['confidence'] < low_conf_threshold])
    
    summary_stats.append({
        'Metric': f'Low Confidence Detections (<{low_conf_threshold})',
        'Value': low_conf_count,
        'Description': f'Number of detections with confidence below {low_conf_threshold}'
    })
    
    summary_stats.append({
        'Metric': 'Low Confidence Percentage',
        'Value': round((low_conf_count / len(df)) * 100, 2) if len(df) > 0 else 0,
        'Description': f'Percentage of detections with confidence below {low_conf_threshold}'
    })
    
    # Create DataFrame and save
    summary_df = pd.DataFrame(summary_stats)
    
    csv_path = output_path / 'detection_summary.csv'
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved summary statistics: {csv_path}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("DETECTION ANALYSIS SUMMARY")
    print("="*60)
    for _, row in summary_df.iterrows():
        print(f"{row['Metric']}: {row['Value']}")
        print(f"  {row['Description']}")
        print()
    
    return summary_df


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze bee detection results and create visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_results.py --csv results/tracking_test_video/test_video_detections.csv
  python analyze_results.py -c detections.csv -o analysis_output/
  python analyze_results.py --csv detections.csv --output detector/analysis/
        """
    )
    
    parser.add_argument(
        "--csv", "-c",
        type=str,
        required=True,
        help="Path to the CSV file containing detection results"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="detector/analysis",
        help="Output directory for plots and summary (default: detector/analysis)"
    )
    
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip creating visualization plots"
    )
    
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip generating summary statistics"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    print("Bee Detection Results Analysis")
    print("=" * 40)
    print(f"CSV file: {args.csv}")
    print(f"Output directory: {args.output}")
    
    # Load detection data
    df = load_detection_data(args.csv)
    
    if len(df) == 0:
        print("No detection data found in CSV file.")
        return
    
    # Create visualizations
    if not args.no_plots:
        print("\nCreating visualizations...")
        create_visualizations(df, args.output)
    
    # Generate summary statistics
    if not args.no_summary:
        print("\nGenerating summary statistics...")
        generate_summary_statistics(df, args.output)
    
    print(f"\nAnalysis complete! Results saved to: {args.output}")


if __name__ == "__main__":
    main()
