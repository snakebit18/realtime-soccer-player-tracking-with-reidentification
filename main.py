"""
Main entry point for the Soccer Player Re-identification Tracker
"""

import argparse
import sys
from pathlib import Path

from soccer_tracker import SoccerReIDTracker
from config import ModelPaths, OUTPUT_DIR


def main():
    """
    Main function to run the soccer player tracker.
    """
    parser = argparse.ArgumentParser(
        description="Soccer Player Re-identification Tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input video.mp4 --output tracked_video.mp4
  python main.py --input video.mp4 --detector custom_yolo.pt --reid custom_reid.pth
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input video file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to output video file (optional)'
    )
    
    parser.add_argument(
        '--detector',
        type=str,
        default=str(ModelPaths.DETECTOR_MODEL),
        help=f'Path to YOLO detector model (default: {ModelPaths.DETECTOR_MODEL})'
    )
    
    parser.add_argument(
        '--reid',
        type=str,
        default=str(ModelPaths.REID_MODEL),
        help=f'Path to re-identification model (default: {ModelPaths.REID_MODEL})'
    )
    
    parser.add_argument(
        '--max-players',
        type=int,
        default=28,
        help='Maximum number of players to track simultaneously (default: 28)'
    )
    
    parser.add_argument(
        '--gallery-size',
        type=int,
        default=10,
        help='Size of embedding gallery per player (default: 10)'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable real-time display window'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input video file not found: {input_path}")
        sys.exit(1)
    
    # Validate model files
    detector_path = Path(args.detector)
    reid_path = Path(args.reid)
    
    if not detector_path.exists():
        print(f"Error: Detector model not found: {detector_path}")
        print("Please download a YOLO model and place it in the models directory.")
        sys.exit(1)
    
    if not reid_path.exists():
        print(f"Error: Re-identification model not found: {reid_path}")
        print("Please download the OSNet model and place it in the models directory.")
        sys.exit(1)
    
    # Setup output path
    output_path = args.output
    if output_path is None:
        output_path = OUTPUT_DIR / f"tracked_{input_path.stem}.mp4"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Update display setting
    if args.no_display:
        from config import VideoConfig
        VideoConfig.DISPLAY_WINDOW = False
    
    print(f"Input video: {input_path}")
    print(f"Output video: {output_path}")
    print(f"Detector model: {detector_path}")
    print(f"Re-ID model: {reid_path}")
    print(f"Max players: {args.max_players}")
    print(f"Gallery size: {args.gallery_size}")
    print("-" * 50)
    
    try:
        # Initialize tracker
        tracker = SoccerReIDTracker(
            detector_model_path=str(detector_path),
            reid_model_path=str(reid_path),
            max_entities=args.max_players,
            embedding_gallery_size=args.gallery_size
        )
        
        # Run tracking
        tracker.run(str(input_path), str(output_path))
        
        # Print final statistics
        stats = tracker.get_statistics()
        print("\nFinal Statistics:")
        print(f"Active players: {stats['active_players']}")
        print(f"Inactive players: {stats['inactive_players']}")
        print(f"Available IDs: {stats['available_ids']}")
        
        print(f"\nTracking completed successfully!")
        print(f"Output saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during tracking: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()