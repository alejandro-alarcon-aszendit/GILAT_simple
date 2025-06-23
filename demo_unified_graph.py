#!/usr/bin/env python3
"""Demo script to test the new unified LangGraph Send API implementation.

This script demonstrates the refactored architecture that replaces ThreadPoolExecutor 
with LangGraph's native Send command for parallel processing.
"""

import time
import asyncio
from typing import List

# Import both old and new implementations for comparison
from src.graphs.summary import MULTI_TOPIC_SUMMARY_GRAPH
from src.graphs.reflection import MULTI_TOPIC_SUMMARY_WITH_REFLECTION_GRAPH
from src.graphs.unified_summary_reflection import UNIFIED_SUMMARY_REFLECTION_GRAPH


def demo_old_vs_new_approach():
    """Compare the old ThreadPoolExecutor approach with new LangGraph Send API."""
    
    # Test data
    test_topics = [
        "machine learning algorithms", 
        "data preprocessing techniques", 
        "model evaluation metrics"
    ]
    test_doc_ids = ["doc_1", "doc_2"]  # Replace with actual doc IDs
    
    print("🔬 DEMO: Comparing Old vs New Parallel Processing Approaches")
    print("=" * 70)
    
    # Test input
    test_input = {
        "topics": test_topics,
        "doc_ids": test_doc_ids,
        "top_k": 5,
        "length": "medium",
        "enable_reflection": True
    }
    
    print(f"📋 Test Configuration:")
    print(f"   Topics: {test_topics}")
    print(f"   Documents: {test_doc_ids}")
    print(f"   Reflection: {test_input['enable_reflection']}")
    print(f"   Length: {test_input['length']}")
    print()
    
    # OLD APPROACH: ThreadPoolExecutor-based reflection graph
    print("🔄 OLD APPROACH: ThreadPoolExecutor + Separate Graphs")
    print("-" * 50)
    
    try:
        start_time = time.time()
        old_result = MULTI_TOPIC_SUMMARY_WITH_REFLECTION_GRAPH.invoke(test_input)
        old_duration = time.time() - start_time
        
        old_summaries = old_result.get("summaries", [])
        old_parallel_info = old_result.get("parallel_processing", {})
        
        print(f"✅ OLD: Completed in {old_duration:.2f}s")
        print(f"   📊 Topics processed: {len(old_summaries)}")
        print(f"   🔍 Reflection applied: {sum(1 for s in old_summaries if s.get('reflection_applied', False))}")
        print(f"   ⚡ Method: {old_parallel_info.get('method', 'Unknown')}")
        print(f"   ⏱️  Parallel time: {old_parallel_info.get('total_time', 0):.2f}s")
        
    except Exception as e:
        print(f"❌ OLD APPROACH FAILED: {str(e)}")
        old_duration = 0
        old_summaries = []
    
    print()
    
    # NEW APPROACH: LangGraph Send API unified graph
    print("🚀 NEW APPROACH: LangGraph Send API + Unified Graph")
    print("-" * 50)
    
    try:
        start_time = time.time()
        new_result = UNIFIED_SUMMARY_REFLECTION_GRAPH.invoke(test_input)
        new_duration = time.time() - start_time
        
        new_summaries = new_result.get("summaries", [])
        new_parallel_info = new_result.get("parallel_processing", {})
        
        print(f"✅ NEW: Completed in {new_duration:.2f}s")
        print(f"   📊 Topics processed: {len(new_summaries)}")
        print(f"   🔍 Reflection applied: {sum(1 for s in new_summaries if s.get('reflection_applied', False))}")
        print(f"   ⚡ Method: {new_parallel_info.get('method', 'Unknown')}")
        print(f"   ⏱️  Parallel time: {new_parallel_info.get('total_time', 0):.2f}s")
        
    except Exception as e:
        print(f"❌ NEW APPROACH FAILED: {str(e)}")
        new_duration = 0
        new_summaries = []
    
    print()
    
    # COMPARISON
    print("📈 COMPARISON & ANALYSIS")
    print("-" * 50)
    
    if old_duration > 0 and new_duration > 0:
        speedup = old_duration / new_duration if new_duration > 0 else 1
        print(f"⏱️  Speed comparison:")
        print(f"   Old approach: {old_duration:.2f}s")
        print(f"   New approach: {new_duration:.2f}s")
        print(f"   Speedup factor: {speedup:.2f}x {'🚀' if speedup > 1 else '🐌' if speedup < 1 else '🟰'}")
        print()
    
    print("🏗️  Architectural improvements:")
    print("   ✅ Eliminated ThreadPoolExecutor dependency")
    print("   ✅ Native LangGraph parallel processing with Send API")
    print("   ✅ Unified graph (summary + reflection integration)")
    print("   ✅ Built-in state management and reducers")
    print("   ✅ Better error handling and state consistency")
    print()
    
    print("🔍 Detailed Results:")
    if old_summaries and new_summaries:
        print(f"   Old summaries: {len(old_summaries)} topics")
        print(f"   New summaries: {len(new_summaries)} topics")
        
        # Show first summary comparison if available
        if old_summaries and new_summaries:
            old_first = old_summaries[0]
            new_first = new_summaries[0]
            print(f"\n   📝 Sample topic: '{old_first.get('topic', 'Unknown')}'")
            print(f"   Old status: {old_first.get('status', 'Unknown')}")
            print(f"   New status: {new_first.get('status', 'Unknown')}")
    
    return {
        "old_duration": old_duration,
        "new_duration": new_duration,
        "old_summaries_count": len(old_summaries) if old_summaries else 0,
        "new_summaries_count": len(new_summaries) if new_summaries else 0
    }


def demo_send_api_features():
    """Demonstrate specific LangGraph Send API features."""
    
    print("\n🎯 DEMO: LangGraph Send API Features")
    print("=" * 70)
    
    # Simple test with minimal topics
    simple_input = {
        "topics": ["artificial intelligence", "machine learning"],
        "doc_ids": ["test_doc"],  # Replace with actual doc ID
        "top_k": 3,
        "length": "short",
        "enable_reflection": False  # Test without reflection first
    }
    
    print("📝 Testing Send API parallel routing...")
    print(f"   Topics: {simple_input['topics']}")
    print(f"   Reflection: {simple_input['enable_reflection']}")
    
    try:
        start_time = time.time()
        result = UNIFIED_SUMMARY_REFLECTION_GRAPH.invoke(simple_input)
        duration = time.time() - start_time
        
        summaries = result.get("summaries", [])
        parallel_info = result.get("parallel_processing", {})
        
        print(f"✅ Send API test completed in {duration:.2f}s")
        print(f"   📊 Parallel method: {parallel_info.get('method', 'Unknown')}")
        print(f"   📈 Topics processed: {len(summaries)}")
        print(f"   ⚡ Avg time per topic: {parallel_info.get('average_time_per_topic', 0):.2f}s")
        
        print("\n🔍 Send API Benefits:")
        print("   ✅ Dynamic parallel routing (no pre-defined thread pool)")
        print("   ✅ State management handled by LangGraph")
        print("   ✅ Built-in error handling and result collection")
        print("   ✅ Native integration with graph execution model")
        print("   ✅ Automatic state reducers for result aggregation")
        
    except Exception as e:
        print(f"❌ Send API test failed: {str(e)}")


def demo_reflection_integration():
    """Test the integrated reflection system within the unified graph."""
    
    print("\n🪞 DEMO: Integrated Reflection System")
    print("=" * 70)
    
    reflection_input = {
        "topics": ["deep learning fundamentals"],
        "doc_ids": ["test_doc"],  # Replace with actual doc ID
        "top_k": 5,
        "length": "medium",
        "enable_reflection": True
    }
    
    print("🔍 Testing integrated reflection with Send API...")
    print(f"   Topic: {reflection_input['topics'][0]}")
    print(f"   Reflection enabled: {reflection_input['enable_reflection']}")
    
    try:
        start_time = time.time()
        result = UNIFIED_SUMMARY_REFLECTION_GRAPH.invoke(reflection_input)
        duration = time.time() - start_time
        
        summaries = result.get("summaries", [])
        reflection_stats = result.get("parallel_processing", {}).get("reflection_statistics", {})
        
        print(f"✅ Reflection test completed in {duration:.2f}s")
        print(f"   📊 Total topics: {reflection_stats.get('total_topics', 0)}")
        print(f"   🔍 Reflection applied: {reflection_stats.get('reflection_applied', 0)}")
        print(f"   ⏭️  Reflection skipped: {reflection_stats.get('reflection_skipped', 0)}")
        
        if summaries:
            first_summary = summaries[0]
            print(f"\n📝 Sample result:")
            print(f"   Topic: {first_summary.get('topic', 'Unknown')}")
            print(f"   Status: {first_summary.get('status', 'Unknown')}")
            print(f"   Reflection applied: {first_summary.get('reflection_applied', False)}")
            
            if first_summary.get('reflection_applied'):
                changes = first_summary.get('changes_made', [])
                print(f"   Changes made: {len(changes)} improvements")
        
        print("\n🏗️  Reflection Integration Benefits:")
        print("   ✅ Reflection runs within same Send API parallel execution")
        print("   ✅ No separate graph invocations (reduced overhead)")
        print("   ✅ Consistent state management across summary + reflection")
        print("   ✅ Atomic processing: each topic gets full treatment in one pass")
        
    except Exception as e:
        print(f"❌ Reflection integration test failed: {str(e)}")


async def main():
    """Run all demos."""
    print("🚀 LangGraph Send API Demo Suite")
    print("=" * 70)
    print("This demo showcases the refactored architecture that replaces")
    print("ThreadPoolExecutor with LangGraph's native Send command.")
    print()
    
    # Run comparison demo
    comparison_results = demo_old_vs_new_approach()
    
    # Run Send API features demo
    demo_send_api_features()
    
    # Run reflection integration demo
    demo_reflection_integration()
    
    print("\n🎉 DEMO COMPLETE!")
    print("=" * 70)
    print("Summary of architectural improvements:")
    print("✅ Replaced ThreadPoolExecutor with LangGraph Send API")
    print("✅ Unified summary + reflection into single graph")
    print("✅ Native parallel processing with state management")
    print("✅ Reduced complexity and better error handling")
    print("✅ True LangGraph-native architecture")


if __name__ == "__main__":
    # Note: You'll need to have documents uploaded to test this properly
    print("⚠️  WARNING: This demo requires actual documents to be uploaded first.")
    print("   Please upload some documents via the API and update doc_ids in the script.")
    print()
    
    # Run the demo
    asyncio.run(main()) 