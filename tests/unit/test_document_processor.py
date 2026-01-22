# docubot/tests/unit/test_document_processor.py

"""
Unit tests for document processor and extractors.
test suite for DocuBot document processing functionality.
"""

import os
import sys
import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

# Try to import document processor and extractors
try:
    from document_processing.processor import DocumentProcessor, get_processor
    HAS_PROCESSOR = True
except ImportError as e:
    print(f"Warning: Could not import DocumentProcessor: {e}")
    HAS_PROCESSOR = False

try:
    from document_processing.extractors.pdf_extractor import PDFExtractor
    HAS_PDF_EXTRACTOR = True
except ImportError as e:
    print(f"Warning: Could not import PDFExtractor: {e}")
    HAS_PDF_EXTRACTOR = False

try:
    from document_processing.extractors.txt_extractor import TXTExtractor
    HAS_TXT_EXTRACTOR = True
except ImportError as e:
    print(f"Warning: Could not import TXTExtractor: {e}")
    HAS_TXT_EXTRACTOR = False


# Skip tests if dependencies are missing
skip_if_no_processor = unittest.skipIf(not HAS_PROCESSOR, "DocumentProcessor not available")
skip_if_no_pdf_extractor = unittest.skipIf(not HAS_PDF_EXTRACTOR, "PDFExtractor not available")
skip_if_no_txt_extractor = unittest.skipIf(not HAS_TXT_EXTRACTOR, "TXTExtractor not available")


class BaseTestDocumentProcessor(unittest.TestCase):
    """Base test class with common setup."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = Path(__file__).parent.parent / "test_data"
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test files if they don't exist
        self._create_test_files()
    
    def _create_test_files(self):
        """Create test files in test_data directory."""
        # Create sample.txt
        txt_file = self.test_data_dir / "sample.txt"
        if not txt_file.exists() or txt_file.stat().st_size == 0:
            txt_content = """This is a sample text document for testing DocuBot.

It contains multiple paragraphs to test text extraction and chunking functionality.

The document processor should be able to extract this text, clean it, and chunk it properly.

This is the final paragraph of the test document."""
            txt_file.write_text(txt_content, encoding='utf-8')
        
        # Create sample.pdf (dummy file)
        pdf_file = self.test_data_dir / "sample.pdf"
        if not pdf_file.exists() or pdf_file.stat().st_size == 0:
            # Create a simple text file with .pdf extension for testing
            # In real scenario, this would be an actual PDF
            pdf_file.write_text("Dummy PDF content for testing", encoding='utf-8')
    
    def tearDown(self):
        """Clean up after tests."""
        # Keep test files for future tests
        pass


@skip_if_no_processor
class TestDocumentProcessorInitialization(BaseTestDocumentProcessor):
    """Test DocumentProcessor initialization and basic properties."""
    
    def test_processor_initialization(self):
        """Test that DocumentProcessor can be initialized."""
        processor = DocumentProcessor()
        self.assertIsNotNone(processor)
        self.assertIsInstance(processor, DocumentProcessor)
    
    def test_default_configuration(self):
        """Test default configuration values."""
        processor = DocumentProcessor()
        stats = processor.get_processing_stats()
        
        # Check required configuration keys
        required_keys = ['chunk_size', 'chunk_overlap', 'max_file_size_mb']
        for key in required_keys:
            self.assertIn(key, stats)
        
        # Check default values
        self.assertEqual(stats['chunk_size'], 500)
        self.assertEqual(stats['chunk_overlap'], 50)
        self.assertEqual(stats['max_file_size_mb'], 100)
    
    def test_custom_configuration(self):
        """Test initialization with custom configuration."""
        custom_config = {
            'chunk_size': 300,
            'chunk_overlap': 25,
            'max_file_size_mb': 50,
            'supported_formats': ['.txt', '.md']
        }
        
        processor = DocumentProcessor(custom_config)
        stats = processor.get_processing_stats()
        
        self.assertEqual(stats['chunk_size'], 300)
        self.assertEqual(stats['chunk_overlap'], 25)
        self.assertEqual(stats['max_file_size_mb'], 50)
    
    def test_singleton_pattern(self):
        """Test get_processor singleton function."""
        processor1 = get_processor()
        processor2 = get_processor()
        
        self.assertIs(processor1, processor2)
        
        # Test with custom config on singleton
        processor3 = get_processor({'chunk_size': 600})
        self.assertIs(processor1, processor3)  # Should still be same instance


@skip_if_no_processor
class TestDocumentProcessorValidation(BaseTestDocumentProcessor):
    """Test file validation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.processor = DocumentProcessor()
    
    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        non_existent = Path("/tmp/nonexistent_file_12345_test.txt")
        result = self.processor.validate_file(non_existent)
        
        self.assertIsInstance(result, dict)
        self.assertFalse(result['is_valid'])
        self.assertGreater(len(result['errors']), 0)
        self.assertIn('File not found', result['errors'][0])
    
    def test_validate_directory_instead_of_file(self):
        """Test validation when path is a directory, not a file."""
        # Use a directory that exists
        directory_path = self.test_data_dir
        
        result = self.processor.validate_file(directory_path)
        
        self.assertFalse(result['is_valid'])
        self.assertIn('Not a file', result['errors'][0])
    
    def test_validate_empty_file(self):
        """Test validation of empty file."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_file = Path(f.name)
        
        try:
            result = self.processor.validate_file(temp_file)
            self.assertFalse(result['is_valid'])
            self.assertIn('empty', result['errors'][0].lower())
        finally:
            if temp_file.exists():
                os.unlink(temp_file)
    
    def test_validate_file_too_large(self):
        """Test validation of file that exceeds size limit."""
        # Create a processor with small size limit
        small_processor = DocumentProcessor({'max_file_size_mb': 0.001})  # 1KB limit
        
        # Create a file larger than limit
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            # Write 2KB of data
            f.write(b'X' * 2048)
            temp_file = Path(f.name)
        
        try:
            result = small_processor.validate_file(temp_file)
            self.assertFalse(result['is_valid'])
            self.assertIn('too large', result['errors'][0].lower())
        finally:
            if temp_file.exists():
                os.unlink(temp_file)
    
    def test_validate_unsupported_format(self):
        """Test validation of unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix='.unsupported', delete=False) as f:
            f.write(b"test content")
            temp_file = Path(f.name)
        
        try:
            result = self.processor.validate_file(temp_file)
            self.assertFalse(result['is_valid'])
            self.assertIn('unsupported', result['errors'][0].lower())
        finally:
            if temp_file.exists():
                os.unlink(temp_file)
    
    def test_validate_valid_text_file(self):
        """Test validation of a valid text file."""
        valid_file = self.test_data_dir / "sample.txt"
        result = self.processor.validate_file(valid_file)
        
        self.assertTrue(result['is_valid'])
        self.assertEqual(len(result['errors']), 0)
        self.assertTrue(result['file_exists'])
        self.assertTrue(result['file_readable'])
        self.assertTrue(result['format_supported'])
        self.assertTrue(result['size_valid'])
    
    def test_validate_file_permission_error(self):
        """Test validation when file is not readable."""
        # Mock os.access to simulate permission error
        with patch('os.access', return_value=False):
            valid_file = self.test_data_dir / "sample.txt"
            result = self.processor.validate_file(valid_file)
            
            # Should fail due to permission error
            self.assertFalse(result['is_valid'])
            self.assertGreater(len(result['errors']), 0)


@skip_if_no_processor
class TestDocumentProcessorFunctionality(BaseTestDocumentProcessor):
    """Test core document processing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        
        # Use a simpler configuration for testing
        self.processor = DocumentProcessor({
            'chunk_size': 100,  # Smaller for testing
            'chunk_overlap': 10,
            'max_file_size_mb': 10
        })
        
        # Mock external dependencies to isolate tests
        self.processor.embedding_service = Mock()
        self.processor.vector_store = Mock()
        self.processor.database = Mock()
        
        # Configure mock returns
        self.processor.embedding_service.generate_embeddings.return_value = [[0.1] * 384]
        self.processor.vector_store.add_documents.return_value = ["vector_id_123"]
        self.processor.database.add_document.return_value = "doc_id_456"
    
    def test_get_supported_formats(self):
        """Test getting supported file formats."""
        formats = self.processor.get_supported_formats()
        
        self.assertIsInstance(formats, list)
        self.assertGreater(len(formats), 0)
        self.assertIn('.txt', formats)
        self.assertIn('.pdf', formats)
    
    def test_get_processing_stats(self):
        """Test getting processing statistics."""
        stats = self.processor.get_processing_stats()
        
        self.assertIsInstance(stats, dict)
        required_keys = [
            'supported_formats', 'chunk_size', 'chunk_overlap',
            'max_file_size_mb', 'ocr_enabled', 'ocr_languages'
        ]
        
        for key in required_keys:
            self.assertIn(key, stats)
    
    @skip_if_no_txt_extractor
    def test_process_valid_text_file(self):
        """Test processing a valid text file."""
        test_file = self.test_data_dir / "sample.txt"
        
        # Mock the extractor
        mock_extractor = Mock()
        mock_extractor.extract.return_value = {
            'text': 'Test extracted text content.',
            'metadata': {'author': 'Test Author', 'title': 'Test Document'}
        }
        
        # Patch the extractor registry
        self.processor.extractor_registry = {'.txt': mock_extractor}
        
        result = self.processor.process_document(test_file)
        
        # Check result structure
        self.assertIsInstance(result, dict)
        required_keys = ['success', 'error', 'document_id', 'chunks_created', 'processing_time']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Should be successful
        self.assertTrue(result['success'])
        self.assertIsNone(result['error'])
        
        # Verify mocks were called
        self.processor.embedding_service.generate_embeddings.assert_called_once()
        self.processor.vector_store.add_documents.assert_called_once()
        self.processor.database.add_document.assert_called_once()
    
    def test_process_nonexistent_file(self):
        """Test processing a non-existent file."""
        non_existent = Path("/tmp/nonexistent_file_test_123.txt")
        
        result = self.processor.process_document(non_existent)
        
        self.assertFalse(result['success'])
        self.assertIsNotNone(result['error'])
        self.assertIn('not found', result['error'].lower())
    
    def test_process_unsupported_format(self):
        """Test processing a file with unsupported format."""
        with tempfile.NamedTemporaryFile(suffix='.unsupported', delete=False) as f:
            f.write(b"test content")
            temp_file = Path(f.name)
        
        try:
            result = self.processor.process_document(temp_file)
            
            self.assertFalse(result['success'])
            self.assertIsNotNone(result['error'])
            self.assertIn('unsupported', result['error'].lower())
        finally:
            if temp_file.exists():
                os.unlink(temp_file)
    
    def test_process_file_extraction_failure(self):
        """Test processing when text extraction fails."""
        test_file = self.test_data_dir / "sample.txt"
        
        # Mock extractor that raises exception
        mock_extractor = Mock()
        mock_extractor.extract.side_effect = Exception("Extraction failed")
        
        self.processor.extractor_registry = {'.txt': mock_extractor}
        
        result = self.processor.process_document(test_file)
        
        self.assertFalse(result['success'])
        self.assertIsNotNone(result['error'])
        self.assertIn('extraction failed', result['error'].lower())


@skip_if_no_processor
class TestDocumentProcessorBatchProcessing(BaseTestDocumentProcessor):
    """Test batch processing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.processor = DocumentProcessor()
        
        # Mock components to speed up tests
        self.processor.embedding_service = Mock()
        self.processor.vector_store = Mock()
        self.processor.database = Mock()
        
        self.processor.embedding_service.generate_embeddings.return_value = [[0.1] * 384]
        self.processor.vector_store.add_documents.return_value = ["vector_id"]
        self.processor.database.add_document.return_value = "doc_id"
    
    def test_batch_processing_empty_list(self):
        """Test batch processing with empty file list."""
        results = self.processor.process_batch([])
        
        self.assertEqual(results, [])
    
    def test_batch_processing_single_file(self):
        """Test batch processing with single file."""
        test_file = self.test_data_dir / "sample.txt"
        
        # Mock extractor
        mock_extractor = Mock()
        mock_extractor.extract.return_value = {
            'text': 'Single file test content.',
            'metadata': {}
        }
        
        self.processor.extractor_registry = {'.txt': mock_extractor}
        
        results = self.processor.process_batch([test_file])
        
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]['success'])
    
    def test_batch_processing_multiple_files(self):
        """Test batch processing with multiple files."""
        # Create temporary test files
        test_files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                f.write(f"Test content for file {i}")
                test_files.append(Path(f.name))
        
        try:
            # Mock extractor
            mock_extractor = Mock()
            mock_extractor.extract.return_value = {
                'text': 'Test content',
                'metadata': {}
            }
            
            self.processor.extractor_registry = {'.txt': mock_extractor}
            
            results = self.processor.process_batch(test_files)
            
            self.assertEqual(len(results), 3)
            
            # All should be successful with mocked dependencies
            for result in results:
                self.assertTrue(result['success'])
        
        finally:
            # Cleanup
            for file_path in test_files:
                if file_path.exists():
                    os.unlink(file_path)
    
    def test_batch_processing_mixed_success_failure(self):
        """Test batch processing with mix of successful and failed files."""
        test_files = []
        
        # Create one valid file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("Valid content")
            test_files.append(Path(f.name))
        
        # Add a non-existent file
        test_files.append(Path("/tmp/nonexistent_batch_test.txt"))
        
        # Add another valid file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("Another valid content")
            test_files.append(Path(f.name))
        
        try:
            # Mock extractor for valid files
            mock_extractor = Mock()
            mock_extractor.extract.return_value = {
                'text': 'Valid content',
                'metadata': {}
            }
            
            self.processor.extractor_registry = {'.txt': mock_extractor}
            
            results = self.processor.process_batch(test_files)
            
            self.assertEqual(len(results), 3)
            
            # Count successes and failures
            successes = sum(1 for r in results if r['success'])
            failures = sum(1 for r in results if not r['success'])
            
            self.assertEqual(successes, 2)  # Two valid files
            self.assertEqual(failures, 1)   # One non-existent file
        
        finally:
            # Cleanup only existing files
            for file_path in test_files:
                if file_path.exists():
                    os.unlink(file_path)


@skip_if_no_pdf_extractor
class TestPDFExtractorIntegration(BaseTestDocumentProcessor):
    """Test PDF extractor integration with processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.processor = DocumentProcessor()
    
    def test_pdf_extractor_in_registry(self):
        """Test that PDF extractor is available in processor registry."""
        # Check if .pdf is in supported formats
        formats = self.processor.get_supported_formats()
        self.assertIn('.pdf', formats)
        
        # Check if extractor is in registry (might be lazy-loaded)
        if hasattr(self.processor, 'extractor_registry'):
            self.assertIn('.pdf', self.processor.extractor_registry)
    
    def test_processor_has_pdf_validation(self):
        """Test that processor can validate PDF files."""
        pdf_file = self.test_data_dir / "sample.pdf"
        
        # Note: Our dummy PDF might not pass real validation
        # This test verifies the method exists and runs without error
        result = self.processor.validate_file(pdf_file)
        
        self.assertIsInstance(result, dict)
        self.assertIn('is_valid', result)
        self.assertIn('errors', result)
        self.assertIn('warnings', result)


class TestDocumentProcessorErrorHandling(BaseTestDocumentProcessor):
    """Test error handling in document processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.processor = DocumentProcessor()
    
    def test_error_handling_in_processing(self):
        """Test that errors are properly caught and returned."""
        test_file = self.test_data_dir / "sample.txt"
        
        # Force an error by removing extractor
        if hasattr(self.processor, 'extractor_registry'):
            self.processor.extractor_registry = {}
        
        result = self.processor.process_document(test_file)
        
        self.assertFalse(result['success'])
        self.assertIsNotNone(result['error'])
        self.assertIn('no extractor', result['error'].lower())
    
    def test_result_structure_on_error(self):
        """Test that result has consistent structure even on error."""
        test_file = Path("/nonexistent/file.txt")
        
        result = self.processor.process_document(test_file)
        
        # Should have all required keys even on error
        required_keys = ['success', 'error', 'document_id', 'processing_time']
        for key in required_keys:
            self.assertIn(key, result)
        
        self.assertFalse(result['success'])
        self.assertIsInstance(result['error'], str)
        self.assertGreater(len(result['error']), 0)


# Integration tests (require actual implementations)
class TestDocumentProcessorIntegration(BaseTestDocumentProcessor):
    """Integration tests for document processor (requires actual implementations)."""
    
    @unittest.skip("Requires full implementation of all dependencies")
    def test_integration_pipeline(self):
        """Test complete integration pipeline (requires all components)."""
        # This test would require actual implementations of:
        # - EmbeddingService
        # - ChromaClient  
        # - SQLiteClient
        # - All extractors
        
        test_file = self.test_data_dir / "sample.txt"
        processor = DocumentProcessor()
        
        result = processor.process_document(test_file)
        
        # Full validation of result
        self.assertTrue(result['success'])
        self.assertGreater(result['chunks_created'], 0)
        self.assertGreater(result['processing_time'], 0)
        self.assertIsNotNone(result['document_id'])
        self.assertIsNotNone(result['vector_ids'])


def run_tests():
    """Run all tests and print summary."""
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    # Capture output
    output = io.StringIO()
    
    with redirect_stdout(output), redirect_stderr(output):
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(sys.modules[__name__])
        runner = unittest.TextTestRunner(verbosity=2, stream=output)
        result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"\n{test}:")
            print(traceback)
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"\n{test}:")
            print(traceback)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run tests when file is executed directly
    success = run_tests()
    sys.exit(0 if success else 1)