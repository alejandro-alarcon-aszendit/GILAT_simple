#!/usr/bin/env python3
"""Test script to verify the modular structure works correctly.

This script tests imports and basic functionality without requiring external services.
"""

def test_imports():
    """Test that all modules can be imported successfully."""
    print("🧪 Testing imports...")
    
    try:
        # Core modules
        from src.core.config import LLMConfig, ParallelConfig, APIConfig
        print("  ✅ Core config imported")
        
        # Models
        from src.models.database import Doc, get_db_session
        from src.models.schemas import SummaryEvaluation, DocOut
        print("  ✅ Models imported")
        
        # Services
        from src.services.document_service import DocumentService
        from src.services.parallel_service import ParallelProcessingService, ParallelWorkload
        print("  ✅ Services imported")
        
        # Graphs
        from src.graphs.ingestion import INGESTION_GRAPH
        from src.graphs.summary import SUMMARY_GRAPH, MULTI_TOPIC_SUMMARY_GRAPH
        from src.graphs.reflection import REFLECTION_GRAPH
        print("  ✅ Graphs imported")
        
        # API
        from src.api.endpoints import DocumentEndpoints, SummaryEndpoints, QAEndpoints
        print("  ✅ API endpoints imported")
        
        # Main app
        from src.main import app, create_app
        print("  ✅ Main app imported")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_configuration():
    """Test that configuration is properly set up."""
    print("\n⚙️  Testing configuration...")
    
    try:
        from src.core.config import LLMConfig, ParallelConfig, APIConfig
        
        # Check LLM config
        assert hasattr(LLMConfig, 'MAIN_LLM')
        assert hasattr(LLMConfig, 'REFLECTION_LLM')
        assert hasattr(LLMConfig, 'IMPROVEMENT_LLM')
        print("  ✅ LLM configuration valid")
        
        # Check parallel config
        assert hasattr(ParallelConfig, 'MAX_TOPIC_WORKERS')
        assert ParallelConfig.MAX_TOPIC_WORKERS > 0
        print("  ✅ Parallel configuration valid")
        
        # Check API config
        assert hasattr(APIConfig, 'TITLE')
        assert APIConfig.VERSION == "2.0"
        print("  ✅ API configuration valid")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False

def test_parallel_service():
    """Test the parallel processing service."""
    print("\n🔄 Testing parallel service...")
    
    try:
        from src.services.parallel_service import ParallelProcessingService, ParallelWorkload
        import time
        
        # Create test workloads
        def test_task(task_id):
            time.sleep(0.1)  # Simulate work
            return f"Task {task_id} completed"
        
        workloads = [
            ParallelWorkload(
                id=f"test_{i}",
                name=f"Test Task {i}",
                function=test_task,
                args=(i,),
                kwargs={}
            )
            for i in range(3)
        ]
        
        # Execute workloads
        result = ParallelProcessingService.execute_workloads(workloads, max_workers=2)
        
        # Verify results
        assert len(result["results"]) == 3
        assert result["performance"]["workloads_count"] == 3
        assert result["performance"]["successful_workloads"] == 3
        print("  ✅ Parallel processing service works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Parallel service test failed: {e}")
        return False

def test_app_creation():
    """Test that the FastAPI app can be created."""
    print("\n🚀 Testing app creation...")
    
    try:
        from src.main import create_app
        
        app = create_app()
        assert app is not None
        assert hasattr(app, 'routes')
        
        # Check that routes exist
        route_paths = [route.path for route in app.routes]
        expected_paths = ["/", "/health", "/documents", "/summary", "/ask"]
        
        for path in expected_paths:
            assert any(path in route_path for route_path in route_paths), f"Missing route: {path}"
        
        print("  ✅ FastAPI app created successfully")
        print(f"  📍 Routes found: {len(route_paths)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ App creation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Modular Document Service v2.0")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_configuration,
        test_parallel_service,
        test_app_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The modular structure is working correctly.")
        print("\n🚀 You can now run the service with:")
        print("   uvicorn src.main:app --reload")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 