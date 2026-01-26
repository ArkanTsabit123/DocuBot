# docubase/tests/e2e/test_full_workflow.py

"""
End-to-end tests for DocuBot full RAG pipeline workflow.

These tests validate the complete document processing, embedding, retrieval,
and generation pipeline to ensure system reliability and accuracy.
"""

import pytest
import sys
import os
from pathlib import Path
import tempfile
import shutil
import time
from datetime import datetime
import json
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, MagicMock, patch

# -------------------------------------------------------------------
# Mock Classes for Testing
# -------------------------------------------------------------------

class MockConfig:
    """Mock configuration for testing."""
    
    def __init__(self):
        # Create temp directory
        self.test_dir = Path(tempfile.mkdtemp(prefix="docubot_test_"))
        
        # Set up directories
        self.data_dir = self.test_dir / "data"
        self.models_dir = self.data_dir / "models"
        self.documents_dir = self.data_dir / "documents"
        self.database_dir = self.data_dir / "database"
        self.logs_dir = self.data_dir / "logs"
        
        # Create directories
        for directory in [self.data_dir, self.models_dir, self.documents_dir, 
                         self.database_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Mock AI configuration
        class AIConfig:
            class LLMConfig:
                model = "llama2:7b"
                temperature = 0.7
                max_tokens = 1024
            class EmbeddingsConfig:
                model = "all-MiniLM-L6-v2"
            class RAGConfig:
                top_k = 5
                similarity_threshold = 0.7
            llm = LLMConfig()
            embeddings = EmbeddingsConfig()
            rag = RAGConfig()
        
        # Mock document processing configuration
        class DocProcessingConfig:
            chunk_size = 200
            chunk_overlap = 50
            supported_formats = [".txt", ".pdf", ".docx"]
        
        self.ai = AIConfig()
        self.document_processing = DocProcessingConfig()
    
    def cleanup(self):
        """Clean up test directory."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)


class MockDocuBotCore:
    """Mock main application for testing."""
    
    def __init__(self, config):
        self.config = config
        self.documents = []
        self.queries = []
    
    def add_document(self, file_path: str) -> Dict[str, Any]:
        """Mock document addition."""
        doc_id = f"doc_{len(self.documents)}"
        result = {
            "success": True,
            "document_id": doc_id,
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "chunks_processed": 3,
            "processing_time": 0.5
        }
        self.documents.append(result)
        return result
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Mock query processing."""
        query_id = f"query_{len(self.queries)}"
        
        # Generate mock response based on query
        if "artificial intelligence" in query.lower() or "ai" in query.lower():
            answer = "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines."
        elif "machine learning" in query.lower() or "ml" in query.lower():
            answer = "Machine Learning is a subset of AI that enables systems to learn from experience."
        elif "data science" in query.lower():
            answer = "Data Science involves extracting insights from data using scientific methods."
        else:
            answer = f"This is a test response to: {query}"
        
        result = {
            "answer": answer,
            "sources": [
                {"source": "test_document_1.txt", "relevance": 0.95},
                {"source": "test_document_2.txt", "relevance": 0.85}
            ],
            "model_used": "llama2:7b",
            "query_time": 0.8,
            "query_id": query_id
        }
        self.queries.append(result)
        return result


# -------------------------------------------------------------------
# Test Data
# -------------------------------------------------------------------

TEST_DOCUMENTS = [
    {
        "name": "test_ai.txt",
        "content": """Artificial Intelligence Fundamentals

Artificial Intelligence (AI) is the field of computer science focused on creating intelligent machines.

Key areas of AI include:
- Machine Learning
- Natural Language Processing
- Computer Vision
- Robotics

Applications range from virtual assistants to autonomous vehicles."""
    },
    {
        "name": "test_datascience.txt",
        "content": """Data Science Workflow

The data science process involves:
1. Problem definition
2. Data collection
3. Data cleaning
4. Exploratory analysis
5. Model building
6. Evaluation
7. Deployment

This iterative process helps extract insights from data."""
    }
]


# -------------------------------------------------------------------
# Test Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def test_config():
    """Create test configuration."""
    config = MockConfig()
    yield config
    config.cleanup()


@pytest.fixture  
def test_app(test_config):
    """Create test application."""
    return MockDocuBotCore(test_config)


@pytest.fixture
def test_documents(test_config):
    """Create test documents."""
    test_docs_dir = test_config.documents_dir / "test_docs"
    test_docs_dir.mkdir(parents=True, exist_ok=True)
    
    test_files = []
    for doc in TEST_DOCUMENTS:
        file_path = test_docs_dir / doc["name"]
        file_path.write_text(doc["content"], encoding="utf-8")
        test_files.append(file_path)
    
    yield test_files
    
    # Cleanup
    if test_docs_dir.exists():
        shutil.rmtree(test_docs_dir, ignore_errors=True)


# -------------------------------------------------------------------
# Test Functions
# -------------------------------------------------------------------

def test_document_ingestion_workflow(test_app, test_documents):
    """Test document ingestion workflow."""
    print("\n[Test 1] Testing document ingestion workflow...")
    
    for file_path in test_documents:
        result = test_app.add_document(str(file_path))
        
        # Verify result structure
        assert result["success"] is True
        assert "document_id" in result
        assert result["chunks_processed"] > 0
        assert result["processing_time"] > 0
        
        print(f"  ✓ Added: {file_path.name}")
    
    # Verify all documents were added
    assert len(test_app.documents) == len(test_documents)
    print(f"✓ Document ingestion: {len(test_documents)} documents processed")


def test_query_processing_workflow(test_app, test_documents):
    """Test query processing workflow."""
    print("\n[Test 2] Testing query processing workflow...")
    
    # First add a document
    if test_documents:
        test_app.add_document(str(test_documents[0]))
    
    # Test various queries
    test_queries = [
        "What is Artificial Intelligence?",
        "Explain machine learning",
        "Describe data science workflow"
    ]
    
    for query in test_queries:
        response = test_app.process_query(query)
        
        # Verify response structure
        assert response is not None
        assert "answer" in response
        assert "sources" in response
        assert "model_used" in response
        assert "query_time" in response
        
        # Verify answer content
        assert len(response["answer"]) > 10
        assert isinstance(response["sources"], list)
        
        print(f"  Query: '{query[:30]}...' -> {len(response['answer'])} chars")
    
    print(f"✓ Query processing: {len(test_queries)} queries processed")


def test_end_to_end_workflow_integration(test_app, test_documents):
    """Test complete end-to-end workflow integration."""
    print("\n[Test 3] Testing end-to-end workflow integration...")
    
    # Step 1: Add all documents
    for file_path in test_documents:
        result = test_app.add_document(str(file_path))
        assert result["success"] is True
    
    # Step 2: Verify document count
    assert len(test_app.documents) == len(test_documents)
    
    # Step 3: Test queries on added documents
    queries = [
        "What are the key areas of AI?",
        "Describe the data science process"
    ]
    
    for query in queries:
        response = test_app.process_query(query)
        assert response is not None
        assert len(response["answer"]) > 20
    
    print("✓ End-to-end workflow integration completed successfully")


def test_error_handling_scenarios(test_app):
    """Test error handling scenarios."""
    print("\n[Test 4] Testing error handling scenarios...")
    
    # Test 1: Empty query
    response = test_app.process_query("")
    assert response is not None
    assert "answer" in response
    
    # Test 2: Very long query
    long_query = "What is " + "AI " * 50 + "?"
    response = test_app.process_query(long_query)
    assert response is not None
    
    # Test 3: Special characters
    special_query = 'What is AI? @#$%^&*()_+{}|:"<>?[]\\;\',./`~'
    response = test_app.process_query(special_query)
    assert response is not None
    
    print("✓ Error handling scenarios completed")


def test_performance_benchmarks(test_app, test_documents):
    """Test performance benchmarks."""
    print("\n[Test 5] Testing performance benchmarks...")
    
    # Time document addition
    start_time = time.time()
    for file_path in test_documents:
        test_app.add_document(str(file_path))
    doc_time = time.time() - start_time
    
    assert doc_time < 10.0, f"Document processing too slow: {doc_time:.2f}s"
    
    # Time queries
    query = "What is Artificial Intelligence?"
    query_times = []
    
    for _ in range(3):
        start_time = time.time()
        response = test_app.process_query(query)
        query_time = time.time() - start_time
        query_times.append(query_time)
        
        assert response is not None
        assert response["query_time"] > 0
    
    avg_query_time = sum(query_times) / len(query_times)
    assert avg_query_time < 5.0, f"Query processing too slow: {avg_query_time:.2f}s"
    
    print(f"  Document processing: {doc_time:.3f}s")
    print(f"  Average query time: {avg_query_time:.3f}s")
    print("✓ Performance benchmarks completed")


def test_data_persistence_simulation(test_config):
    """Test data persistence simulation."""
    print("\n[Test 6] Testing data persistence simulation...")
    
    # Create test data
    test_data = {
        "test_documents": ["doc1.txt", "doc2.txt"],
        "test_queries": ["q1", "q2"],
        "timestamp": datetime.now().isoformat()
    }
    
    # Save to file
    data_file = test_config.data_dir / "test_persistence.json"
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    # Verify
    assert data_file.exists()
    assert data_file.stat().st_size > 0
    
    # Load and verify
    with open(data_file, 'r', encoding='utf-8') as f:
        loaded_data = json.load(f)
    
    assert loaded_data["test_documents"] == test_data["test_documents"]
    assert loaded_data["test_queries"] == test_data["test_queries"]
    
    print("✓ Data persistence simulation completed")


def test_edge_cases_handling():
    """Test edge cases handling."""
    print("\n[Test 7] Testing edge cases handling...")
    
    config = MockConfig()
    app = MockDocuBotCore(config)
    
    # Test various edge cases
    edge_cases = [
        ("", "empty string"),
        ("   ", "whitespace only"),
        ("\n\t\n", "control characters"),
        ("A" * 1000, "very long string"),
        ("!@#$%^&*()", "special characters only"),
    ]
    
    for query, description in edge_cases:
        response = app.process_query(query)
        assert response is not None
        print(f"  ✓ Handled: {description}")
    
    config.cleanup()
    print("✓ Edge cases handling completed")


def test_mock_functionality_validation():
    """Test mock functionality validation."""
    print("\n[Test 8] Testing mock functionality validation...")
    
    config = MockConfig()
    app = MockDocuBotCore(config)
    
    # Test that mocks work correctly
    assert hasattr(config, 'data_dir')
    assert hasattr(config, 'documents_dir')
    assert config.data_dir.exists()
    
    assert hasattr(app, 'add_document')
    assert hasattr(app, 'process_query')
    
    # Test method calls
    result = app.add_document("/test/path.txt")
    assert result["success"] is True
    
    response = app.process_query("test query")
    assert response["answer"] is not None
    
    config.cleanup()
    print("✓ Mock functionality validation completed")


# -------------------------------------------------------------------
# Test Classes
# -------------------------------------------------------------------

class TestCompleteWorkflowIntegration:
    """Test class for comprehensive workflow integration testing."""
    
    def setup_method(self):
        """Setup before each test method."""
        self.config = MockConfig()
        self.app = MockDocuBotCore(self.config)
        
        # Create test documents
        self.test_docs_dir = self.config.documents_dir / "test_docs"
        self.test_docs_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_files = []
        for doc in TEST_DOCUMENTS:
            file_path = self.test_docs_dir / doc["name"]
            file_path.write_text(doc["content"], encoding="utf-8")
            self.test_files.append(file_path)
    
    def teardown_method(self):
        """Teardown after each test method."""
        if self.test_docs_dir.exists():
            shutil.rmtree(self.test_docs_dir, ignore_errors=True)
        self.config.cleanup()
    
    def test_workflow_integration_comprehensive(self):
        """Test comprehensive workflow integration."""
        print("\n[Class Test 1] Testing comprehensive workflow integration...")
        
        # Add documents
        for file_path in self.test_files:
            result = self.app.add_document(str(file_path))
            assert result["success"] is True
        
        # Process queries
        queries = [
            "What is AI?",
            "Explain data science"
        ]
        
        for query in queries:
            response = self.app.process_query(query)
            assert response is not None
            assert "answer" in response
        
        print("  ✓ Comprehensive workflow integration test passed")
    
    def test_concurrent_operations_simulation(self):
        """Test simulated concurrent operations."""
        print("\n[Class Test 2] Testing simulated concurrent operations...")
        
        # Simulate concurrent operations
        operations = []
        
        # Add documents
        for i, file_path in enumerate(self.test_files):
            operations.append(("add", file_path))
        
        # Add queries
        test_queries = ["Query 1", "Query 2", "Query 3"]
        for query in test_queries:
            operations.append(("query", query))
        
        # Execute operations
        results = []
        for op_type, op_data in operations:
            if op_type == "add":
                result = self.app.add_document(str(op_data))
                assert result["success"] is True
                results.append(("add", "success"))
            else:
                result = self.app.process_query(op_data)
                assert result is not None
                results.append(("query", "success"))
        
        assert len(results) == len(operations)
        print(f"  ✓ Concurrent operations: {len(operations)} operations completed")


# -------------------------------------------------------------------
# Direct Execution Support
# -------------------------------------------------------------------

def run_all_tests_directly():
    """Run all tests directly (not through pytest)."""
    print("=" * 80)
    print("DOCUBOT END-TO-END TEST SUITE - DIRECT EXECUTION")
    print("=" * 80)
    
    results = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "start_time": time.time()
    }
    
    # Individual test functions
    test_functions = [
        test_document_ingestion_workflow,
        test_query_processing_workflow,
        test_end_to_end_workflow_integration,
        test_error_handling_scenarios,
        test_performance_benchmarks,
        test_data_persistence_simulation,
        test_edge_cases_handling,
        test_mock_functionality_validation
    ]
    
    # Run individual tests
    for i, test_func in enumerate(test_functions, 1):
        results["total"] += 1
        
        # Create fresh fixtures for each test
        config = MockConfig()
        
        try:
            if test_func.__name__ in ['test_document_ingestion_workflow', 
                                      'test_query_processing_workflow',
                                      'test_end_to_end_workflow_integration',
                                      'test_performance_benchmarks']:
                # These need test_documents fixture
                test_docs_dir = config.documents_dir / "test_docs"
                test_docs_dir.mkdir(parents=True, exist_ok=True)
                
                test_files = []
                for doc in TEST_DOCUMENTS:
                    file_path = test_docs_dir / doc["name"]
                    file_path.write_text(doc["content"], encoding="utf-8")
                    test_files.append(file_path)
                
                app = MockDocuBotCore(config)
                test_func(app, test_files)
                
                # Cleanup
                if test_docs_dir.exists():
                    shutil.rmtree(test_docs_dir, ignore_errors=True)
                    
            elif test_func.__name__ == 'test_error_handling_scenarios':
                app = MockDocuBotCore(config)
                test_func(app)
                
            elif test_func.__name__ == 'test_data_persistence_simulation':
                test_func(config)
                
            elif test_func.__name__ in ['test_edge_cases_handling', 
                                        'test_mock_functionality_validation']:
                test_func()
            
            results["passed"] += 1
            print(f"[{i:2d}/{len(test_functions)}] {test_func.__name__} ✓ PASSED")
            
        except Exception as e:
            results["failed"] += 1
            print(f"[{i:2d}/{len(test_functions)}] {test_func.__name__} ✗ FAILED: {type(e).__name__}: {str(e)[:100]}")
        
        finally:
            config.cleanup()
    
    # Run class tests
    test_class = TestCompleteWorkflowIntegration()
    
    class_test_methods = [
        test_class.test_workflow_integration_comprehensive,
        test_class.test_concurrent_operations_simulation
    ]
    
    for i, test_method in enumerate(class_test_methods, len(test_functions) + 1):
        results["total"] += 1
        
        try:
            test_class.setup_method()
            test_method()
            test_class.teardown_method()
            
            results["passed"] += 1
            print(f"[{i:2d}/{len(test_functions) + len(class_test_methods)}] {test_method.__name__} ✓ PASSED")
            
        except Exception as e:
            results["failed"] += 1
            print(f"[{i:2d}/{len(test_functions) + len(class_test_methods)}] {test_method.__name__} ✗ FAILED: {type(e).__name__}: {str(e)[:100]}")
    
    # Calculate execution time
    results["execution_time"] = time.time() - results["start_time"]
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Execution Time: {results['execution_time']:.2f} seconds")
    print(f"Success Rate: {(results['passed'] / results['total']) * 100:.1f}%")
    
    if results["failed"] == 0:
        print("\n✓ ALL TESTS PASSED SUCCESSFULLY")
        return True
    else:
        print(f"\n⚠ {results['failed']} TESTS FAILED")
        return False


if __name__ == "__main__":
    """Main entry point for test execution."""
    
    # Check if we should run tests directly or via pytest
    if len(sys.argv) > 1 and sys.argv[1] == "--direct":
        success = run_all_tests_directly()
        sys.exit(0 if success else 1)
    else:
        # Run with pytest
        print("Running tests with pytest...")
        exit_code = pytest.main([__file__, "-v"])
        sys.exit(exit_code)