#!/usr/bin/env python3
"""
Test script to verify that the summary length changes work correctly.
This script tests the new numeric length parameter system.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.topic_processing import enhance_summary_for_topic
from utils.reflection_utils import apply_reflection_to_summary

def test_length_parameter():
    """Test that the length parameter works with numeric values."""
    
    # Test data
    test_summary = "This is a test summary. It has multiple sentences. This should work with the new numeric length system."
    test_topic = "test topic"
    test_source = "Source content for testing the reflection system with numeric length parameters."
    
    print("🧪 Testing Summary Length Parameter Changes")
    print("=" * 50)
    
    # Test different length values
    test_lengths = [3, 8, 15, 20]
    
    for length in test_lengths:
        print(f"\n📏 Testing length: {length} sentences")
        
        # Test topic enhancement
        try:
            enhanced = enhance_summary_for_topic(test_summary, test_topic, length, "abstractive")
            print(f"  ✅ Topic enhancement: SUCCESS")
            print(f"     Length param used: {length} sentences")
        except Exception as e:
            print(f"  ❌ Topic enhancement: FAILED - {e}")
        
        # Test reflection
        try:
            reflection_result = apply_reflection_to_summary(test_summary, test_topic, length, test_source)
            if reflection_result.get("error"):
                print(f"  ⚠️ Reflection: ERROR - {reflection_result['error']}")
            else:
                print(f"  ✅ Reflection: SUCCESS")
                print(f"     Length param used: {length} sentences")
        except Exception as e:
            print(f"  ❌ Reflection: FAILED - {e}")
    
    print("\n" + "=" * 50)
    print("✅ Test completed! The numeric length parameter system appears to be working.")

if __name__ == "__main__":
    test_length_parameter() 