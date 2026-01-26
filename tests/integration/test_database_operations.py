"""
Database Integration Tests for DocuBot
Advanced database operations, edge cases, and data integrity verification.
"""

import pytest
import sys
import os
import tempfile
import sqlite3
import threading
import shutil
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from database.sqlite_client import SQLiteClient, DatabaseError, DatabaseConnectionError, DatabaseQueryError
    SQLITE_CLIENT_AVAILABLE = True
except ImportError:
    SQLITE_CLIENT_AVAILABLE = False


class TemporaryDatabaseManager:
    """Manages temporary database files for testing."""
    
    def __init__(self):
        self.temp_files = []
    
    def create_temp_database(self) -> str:
        """Create a temporary database file."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_file.close()
        self.temp_files.append(temp_file.name)
        return temp_file.name
    
    def cleanup(self):
        """Remove all temporary database files."""
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass
        self.temp_files.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


@pytest.mark.skipif(not SQLITE_CLIENT_AVAILABLE, reason="SQLiteClient module not available")
class TestDataIntegrityValidation:
    """Validate data integrity and constraint enforcement mechanisms."""
    
    def setup_method(self):
        self.db_manager = TemporaryDatabaseManager()
    
    def teardown_method(self):
        self.db_manager.cleanup()
    
    def test_foreign_key_constraint_enforcement(self):
        """Verify foreign key constraints maintain referential integrity."""
        database_path = self.db_manager.create_temp_database()
        client = SQLiteClient(database_path)
        
        client.execute_query("PRAGMA foreign_keys = ON")
        
        client.execute_query("""
            CREATE TABLE departments (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE
            )
        """)
        
        client.execute_query("""
            CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                department_id INTEGER,
                FOREIGN KEY (department_id) 
                REFERENCES departments(id) 
                ON DELETE SET NULL
            )
        """)
        
        client.execute_query(
            "INSERT INTO departments (name) VALUES (?)",
            ("Engineering",)
        )
        
        client.execute_query(
            "INSERT INTO employees (name, department_id) VALUES (?, ?)",
            ("John Doe", 1)
        )
        
        with pytest.raises(sqlite3.IntegrityError):
            client.execute_query(
                "INSERT INTO employees (name, department_id) VALUES (?, ?)",
                ("Jane Smith", 999)
            )
        
        client.close_connection()
    
    def test_data_type_constraint_validation(self):
        """Ensure data type validation and custom constraints are properly enforced."""
        database_path = self.db_manager.create_temp_database()
        client = SQLiteClient(database_path)
        
        client.execute_query("""
            CREATE TABLE validated_data (
                id INTEGER PRIMARY KEY,
                email TEXT UNIQUE,
                age INTEGER CHECK (age >= 0 AND age <= 150),
                score REAL CHECK (score >= 0.0 AND score <= 100.0),
                status TEXT CHECK (status IN ('active', 'inactive', 'pending'))
            )
        """)
        
        valid_data = ('test@example.com', 30, 85.5, 'active')
        client.execute_query("""
            INSERT INTO validated_data (email, age, score, status)
            VALUES (?, ?, ?, ?)
        """, valid_data)
        
        with pytest.raises(sqlite3.IntegrityError):
            client.execute_query("""
                INSERT INTO validated_data (email, age, score, status)
                VALUES (?, ?, ?, ?)
            """, ('invalid@example.com', 200, 85.5, 'active'))
        
        client.close_connection()
    
    def test_not_null_constraint(self):
        """Test NOT NULL constraint enforcement."""
        database_path = self.db_manager.create_temp_database()
        client = SQLiteClient(database_path)
        
        client.execute_query("""
            CREATE TABLE not_null_test (
                id INTEGER PRIMARY KEY,
                required_field TEXT NOT NULL
            )
        """)
        
        with pytest.raises(sqlite3.IntegrityError):
            client.execute_query(
                "INSERT INTO not_null_test (required_field) VALUES (?)",
                (None,)
            )
        
        client.close_connection()
    
    def test_unique_constraint(self):
        """Test UNIQUE constraint enforcement."""
        database_path = self.db_manager.create_temp_database()
        client = SQLiteClient(database_path)
        
        client.execute_query("""
            CREATE TABLE unique_test (
                id INTEGER PRIMARY KEY,
                code TEXT UNIQUE
            )
        """)
        
        client.execute_query(
            "INSERT INTO unique_test (code) VALUES (?)",
            ("ABC123",)
        )
        
        with pytest.raises(sqlite3.IntegrityError):
            client.execute_query(
                "INSERT INTO unique_test (code) VALUES (?)",
                ("ABC123",)
            )
        
        client.close_connection()


@pytest.mark.skipif(not SQLITE_CLIENT_AVAILABLE, reason="SQLiteClient module not available")
class TestAdvancedQueryOperations:
    """Evaluate complex SQL operations and advanced query patterns."""
    
    def setup_method(self):
        self.db_manager = TemporaryDatabaseManager()
    
    def teardown_method(self):
        self.db_manager.cleanup()
    
    def test_join_operations_with_aggregations(self):
        """Validate complex join operations combined with aggregate functions."""
        database_path = self.db_manager.create_temp_database()
        client = SQLiteClient(database_path)
        
        client.execute_query("""
            CREATE TABLE categories (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            )
        """)
        
        client.execute_query("""
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                category_id INTEGER,
                price REAL,
                FOREIGN KEY (category_id) REFERENCES categories(id)
            )
        """)
        
        test_categories = [("Electronics",), ("Clothing",), ("Books",)]
        for category in test_categories:
            client.execute_query(
                "INSERT INTO categories (name) VALUES (?)",
                category
            )
        
        test_products = [
            ("Laptop", 1, 999.99),
            ("T-Shirt", 2, 19.99),
            ("Python Book", 3, 49.99),
            ("Smartphone", 1, 699.99)
        ]
        for product in test_products:
            client.execute_query(
                "INSERT INTO products (name, category_id, price) VALUES (?, ?, ?)",
                product
            )
        
        result = client.execute_query("""
            SELECT 
                c.name AS category_name,
                COUNT(p.id) AS product_count,
                AVG(p.price) AS average_price
            FROM categories c
            LEFT JOIN products p ON c.id = p.category_id
            GROUP BY c.id, c.name
            ORDER BY c.name
        """)
        
        result_rows = result.fetchall()
        
        assert len(result_rows) == 3
        
        electronics_data = [row for row in result_rows if row[0] == "Electronics"][0]
        assert electronics_data[1] == 2
        assert 800.0 < electronics_data[2] < 900.0
        
        client.close_connection()
    
    def test_common_table_expressions_and_subqueries(self):
        """Test Common Table Expressions and subquery operations."""
        database_path = self.db_manager.create_temp_database()
        client = SQLiteClient(database_path)
        
        client.execute_query("""
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                customer_id INTEGER,
                amount REAL,
                order_date TEXT
            )
        """)
        
        orders_test_data = [
            (1, 100.0, '2024-01-15'),
            (1, 200.0, '2024-01-20'),
            (2, 150.0, '2024-01-18'),
            (2, 300.0, '2024-01-25'),
            (3, 50.0, '2024-01-10')
        ]
        
        for order in orders_test_data:
            client.execute_query(
                "INSERT INTO orders (customer_id, amount, order_date) VALUES (?, ?, ?)",
                order
            )
        
        result = client.execute_query("""
            WITH customer_totals AS (
                SELECT 
                    customer_id,
                    COUNT(*) as order_count,
                    SUM(amount) as total_amount
                FROM orders
                GROUP BY customer_id
            )
            SELECT customer_id, order_count, total_amount
            FROM customer_totals
            WHERE total_amount > 100
            ORDER BY total_amount DESC
        """)
        
        result_rows = result.fetchall()
        
        assert len(result_rows) == 2
        
        assert result_rows[0][0] == 2
        assert result_rows[0][2] == 450.0
        
        assert result_rows[1][0] == 1
        assert result_rows[1][2] == 300.0
        
        client.close_connection()
    
    def test_window_functions(self):
        """Test window function operations."""
        database_path = self.db_manager.create_temp_database()
        client = SQLiteClient(database_path)
        
        client.execute_query("""
            CREATE TABLE sales (
                id INTEGER PRIMARY KEY,
                region TEXT,
                salesperson TEXT,
                amount REAL,
                sale_date TEXT
            )
        """)
        
        sales_data = [
            ("North", "Alice", 1000.0, "2024-01-15"),
            ("North", "Bob", 1500.0, "2024-01-16"),
            ("South", "Charlie", 800.0, "2024-01-15"),
            ("South", "Alice", 1200.0, "2024-01-17"),
            ("North", "Charlie", 900.0, "2024-01-18"),
        ]
        
        for sale in sales_data:
            client.execute_query(
                "INSERT INTO sales (region, salesperson, amount, sale_date) VALUES (?, ?, ?, ?)",
                sale
            )
        
        result = client.execute_query("""
            SELECT 
                region,
                salesperson,
                amount,
                RANK() OVER (PARTITION BY region ORDER BY amount DESC) as rank_in_region
            FROM sales
            ORDER BY region, rank_in_region
        """)
        
        result_rows = result.fetchall()
        
        assert len(result_rows) == 5
        
        north_region_rows = [row for row in result_rows if row[0] == "North"]
        assert len(north_region_rows) == 3
        
        top_north_salesperson = north_region_rows[0][1]
        assert top_north_salesperson == "Bob"
        
        client.close_connection()


@pytest.mark.skipif(not SQLITE_CLIENT_AVAILABLE, reason="SQLiteClient module not available")
class TestTransactionManagement:
    """Test database transaction management and ACID properties."""
    
    def setup_method(self):
        self.db_manager = TemporaryDatabaseManager()
    
    def teardown_method(self):
        self.db_manager.cleanup()
    
    def test_transaction_commit_operation(self):
        """Verify successful transaction commit operations."""
        database_path = self.db_manager.create_temp_database()
        client = SQLiteClient(database_path)
        
        client.execute_query("""
            CREATE TABLE transaction_test (
                id INTEGER PRIMARY KEY,
                value INTEGER
            )
        """)
        
        client.begin_transaction()
        
        client.execute_query(
            "INSERT INTO transaction_test (value) VALUES (?)",
            (100,)
        )
        
        client.commit_transaction()
        
        result = client.execute_query("SELECT COUNT(*) FROM transaction_test")
        count = result.fetchone()[0]
        
        assert count == 1
        
        client.close_connection()
    
    def test_transaction_rollback_operation(self):
        """Verify transaction rollback on integrity constraint violation."""
        database_path = self.db_manager.create_temp_database()
        client = SQLiteClient(database_path)
        
        client.execute_query("""
            CREATE TABLE inventory (
                id INTEGER PRIMARY KEY,
                item_name TEXT UNIQUE,
                quantity INTEGER
            )
        """)
        
        client.begin_transaction()
        
        client.execute_query(
            "INSERT INTO inventory (item_name, quantity) VALUES (?, ?)",
            ("Item A", 10)
        )
        
        try:
            client.execute_query(
                "INSERT INTO inventory (item_name, quantity) VALUES (?, ?)",
                ("Item A", 20)
            )
            pytest.fail("Expected integrity constraint violation")
        except sqlite3.IntegrityError:
            client.rollback_transaction()
        
        result = client.execute_query("SELECT COUNT(*) FROM inventory")
        count = result.fetchone()[0]
        
        assert count == 0
        
        client.close_connection()
    
    def test_nested_transaction_context_manager(self):
        """Test transaction management using context manager."""
        database_path = self.db_manager.create_temp_database()
        client = SQLiteClient(database_path)
        
        client.execute_query("""
            CREATE TABLE context_test (
                id INTEGER PRIMARY KEY,
                data TEXT
            )
        """)
        
        try:
            client.begin_transaction()
            
            client.execute_query(
                "INSERT INTO context_test (data) VALUES (?)",
                ("Test Data 1",)
            )
            
            client.execute_query(
                "INSERT INTO context_test (data) VALUES (?)",
                ("Test Data 2",)
            )
            
            client.commit_transaction()
            
        except Exception:
            client.rollback_transaction()
            raise
        
        result = client.execute_query("SELECT COUNT(*) FROM context_test")
        count = result.fetchone()[0]
        
        assert count == 2
        
        client.close_connection()


@pytest.mark.skipif(not SQLITE_CLIENT_AVAILABLE, reason="SQLiteClient module not available")
class TestConcurrentOperations:
    """Test concurrent database access patterns."""
    
    def setup_method(self):
        self.db_manager = TemporaryDatabaseManager()
    
    def teardown_method(self):
        self.db_manager.cleanup()
    
    def test_concurrent_read_operations(self):
        """Verify multiple concurrent read operations function correctly."""
        database_path = self.db_manager.create_temp_database()
        
        setup_connection = sqlite3.connect(database_path)
        setup_connection.execute("""
            CREATE TABLE concurrent_test (
                id INTEGER PRIMARY KEY,
                value INTEGER
            )
        """)
        
        for i in range(10):
            setup_connection.execute(
                "INSERT INTO concurrent_test (value) VALUES (?)",
                (i,)
            )
        
        setup_connection.commit()
        setup_connection.close()
        
        read_results = []
        read_lock = threading.Lock()
        
        def concurrent_reader(thread_identifier):
            connection = sqlite3.connect(database_path, timeout=10.0)
            cursor = connection.execute("SELECT COUNT(*) FROM concurrent_test")
            count = cursor.fetchone()[0]
            connection.close()
            
            with read_lock:
                read_results.append((thread_identifier, count))
        
        reader_threads = []
        for thread_id in range(5):
            thread = threading.Thread(target=concurrent_reader, args=(thread_id,))
            reader_threads.append(thread)
            thread.start()
        
        for thread in reader_threads:
            thread.join()
        
        assert len(read_results) == 5
        for _, count_value in read_results:
            assert count_value == 10
    
    def test_concurrent_writes_with_isolation(self):
        """Test concurrent write operations with proper isolation."""
        database_path = self.db_manager.create_temp_database()
        
        connection = sqlite3.connect(database_path)
        connection.execute("""
            CREATE TABLE concurrent_writes (
                id INTEGER PRIMARY KEY,
                thread_id INTEGER,
                data TEXT
            )
        """)
        connection.close()
        
        write_results = []
        write_lock = threading.Lock()
        
        def concurrent_writer(thread_id):
            try:
                connection = sqlite3.connect(database_path, timeout=5.0)
                connection.execute("BEGIN IMMEDIATE")
                
                connection.execute(
                    "INSERT INTO concurrent_writes (thread_id, data) VALUES (?, ?)",
                    (thread_id, f"Data from thread {thread_id}")
                )
                
                connection.commit()
                connection.close()
                
                with write_lock:
                    write_results.append((thread_id, "success"))
                    
            except sqlite3.OperationalError as e:
                with write_lock:
                    write_results.append((thread_id, f"failed: {str(e)}"))
        
        writer_threads = []
        for thread_id in range(3):
            thread = threading.Thread(target=concurrent_writer, args=(thread_id,))
            writer_threads.append(thread)
            thread.start()
        
        for thread in writer_threads:
            thread.join()
        
        connection = sqlite3.connect(database_path)
        cursor = connection.execute("SELECT COUNT(*) FROM concurrent_writes")
        total_rows = cursor.fetchone()[0]
        connection.close()
        
        assert total_rows == 3


@pytest.mark.skipif(not SQLITE_CLIENT_AVAILABLE, reason="SQLiteClient module not available")
class TestDatabaseBackupOperations:
    """Test database backup and restore functionality."""
    
    def setup_method(self):
        self.db_manager = TemporaryDatabaseManager()
    
    def teardown_method(self):
        self.db_manager.cleanup()
    
    def test_database_backup_functionality(self):
        """Verify database backup operation creates functional copy."""
        source_path = self.db_manager.create_temp_database()
        backup_path = self.db_manager.create_temp_database()
        
        source_connection = sqlite3.connect(source_path)
        source_connection.execute("""
            CREATE TABLE backup_test (
                id INTEGER PRIMARY KEY,
                data TEXT
            )
        """)
        
        for i in range(5):
            source_connection.execute(
                "INSERT INTO backup_test (data) VALUES (?)",
                (f"Test Data {i}",)
            )
        
        source_connection.commit()
        source_connection.close()
        
        shutil.copy2(source_path, backup_path)
        
        backup_connection = sqlite3.connect(backup_path)
        backup_cursor = backup_connection.execute("SELECT COUNT(*) FROM backup_test")
        backup_count = backup_cursor.fetchone()[0]
        backup_connection.close()
        
        assert backup_count == 5
    
    def test_database_restore_functionality(self):
        """Test database restoration from backup."""
        source_path = self.db_manager.create_temp_database()
        backup_path = self.db_manager.create_temp_database()
        restore_path = self.db_manager.create_temp_database()
        
        source_connection = sqlite3.connect(source_path)
        source_connection.execute("""
            CREATE TABLE restore_test (
                id INTEGER PRIMARY KEY,
                critical_data TEXT
            )
        """)
        
        source_connection.execute(
            "INSERT INTO restore_test (critical_data) VALUES (?)",
            ("Business Critical Data",)
        )
        
        source_connection.commit()
        source_connection.close()
        
        shutil.copy2(source_path, backup_path)
        
        shutil.copy2(backup_path, restore_path)
        
        restore_connection = sqlite3.connect(restore_path)
        restore_cursor = restore_connection.execute(
            "SELECT critical_data FROM restore_test"
        )
        restored_data = restore_cursor.fetchone()[0]
        restore_connection.close()
        
        assert restored_data == "Business Critical Data"


@pytest.mark.skipif(not SQLITE_CLIENT_AVAILABLE, reason="SQLiteClient module not available")
class TestEdgeCaseScenarios:
    """Test edge cases and boundary conditions."""
    
    def setup_method(self):
        self.db_manager = TemporaryDatabaseManager()
    
    def teardown_method(self):
        self.db_manager.cleanup()
    
    def test_large_data_volume_handling(self):
        """Verify database handles large data volumes correctly."""
        database_path = self.db_manager.create_temp_database()
        client = SQLiteClient(database_path)
        
        client.execute_query("""
            CREATE TABLE large_data (
                id INTEGER PRIMARY KEY,
                content TEXT
            )
        """)
        
        large_content = "X" * 10000
        
        client.begin_transaction()
        
        for record_id in range(100):
            client.execute_query(
                "INSERT INTO large_data (content) VALUES (?)",
                (f"{large_content}_{record_id}",)
            )
        
        client.commit_transaction()
        
        result = client.execute_query("SELECT COUNT(*) FROM large_data")
        count = result.fetchone()[0]
        
        assert count == 100
        
        result = client.execute_query("""
            SELECT LENGTH(content), COUNT(*) 
            FROM large_data 
            GROUP BY LENGTH(content)
        """)
        
        row = result.fetchone()
        assert row[0] == 10005
        
        client.close_connection()
    
    def test_special_character_handling(self):
        """Verify database correctly handles special characters and SQL injection attempts."""
        database_path = self.db_manager.create_temp_database()
        client = SQLiteClient(database_path)
        
        client.execute_query("""
            CREATE TABLE special_chars (
                id INTEGER PRIMARY KEY,
                text_content TEXT
            )
        """)
        
        test_strings = [
            "Normal text",
            "Text with 'quotes'",
            'Text with "double quotes"',
            "Text with \\backslashes\\",
            "Unicode: Café, naïve, résumé",
            "Emoji: 🚀 📚 💻",
            "SQL injection attempt: ' OR '1'='1",
        ]
        
        for text_value in test_strings:
            client.execute_query(
                "INSERT INTO special_chars (text_content) VALUES (?)",
                (text_value,)
            )
        
        result = client.execute_query("SELECT COUNT(*) FROM special_chars")
        count = result.fetchone()[0]
        
        assert count == len(test_strings)
        
        client.close_connection()
    
    def test_empty_database_operations(self):
        """Test operations on empty databases and tables."""
        database_path = self.db_manager.create_temp_database()
        client = SQLiteClient(database_path)
        
        client.execute_query("""
            CREATE TABLE empty_table (
                id INTEGER PRIMARY KEY,
                data TEXT
            )
        """)
        
        result = client.execute_query("SELECT COUNT(*) FROM empty_table")
        count = result.fetchone()[0]
        
        assert count == 0
        
        result = client.execute_query("SELECT * FROM empty_table")
        rows = result.fetchall()
        
        assert len(rows) == 0
        
        client.close_connection()
    
    def test_maximum_column_length_handling(self):
        """Test handling of maximum column lengths."""
        database_path = self.db_manager.create_temp_database()
        client = SQLiteClient(database_path)
        
        client.execute_query("""
            CREATE TABLE max_length_test (
                id INTEGER PRIMARY KEY,
                long_text TEXT
            )
        """)
        
        very_long_text = "A" * 1000000
        
        client.execute_query(
            "INSERT INTO max_length_test (long_text) VALUES (?)",
            (very_long_text,)
        )
        
        result = client.execute_query("SELECT LENGTH(long_text) FROM max_length_test")
        length = result.fetchone()[0]
        
        assert length == 1000000
        
        client.close_connection()


@pytest.mark.skipif(not SQLITE_CLIENT_AVAILABLE, reason="SQLiteClient module not available")
class TestPerformanceMetrics:
    """Test database performance characteristics."""
    
    def setup_method(self):
        self.db_manager = TemporaryDatabaseManager()
        self.performance_threshold = 2.0
    
    def teardown_method(self):
        self.db_manager.cleanup()
    
    def test_bulk_insert_performance(self):
        """Test performance of bulk insert operations."""
        database_path = self.db_manager.create_temp_database()
        client = SQLiteClient(database_path)
        
        client.execute_query("""
            CREATE TABLE performance_test (
                id INTEGER PRIMARY KEY,
                data TEXT,
                timestamp TEXT
            )
        """)
        
        start_time = time.time()
        
        client.begin_transaction()
        
        for i in range(1000):
            client.execute_query(
                "INSERT INTO performance_test (data, timestamp) VALUES (?, ?)",
                (f"Data {i}", datetime.now().isoformat())
            )
        
        client.commit_transaction()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        result = client.execute_query("SELECT COUNT(*) FROM performance_test")
        count = result.fetchone()[0]
        
        assert count == 1000
        assert elapsed_time < self.performance_threshold
        
        client.close_connection()
    
    def test_indexed_query_performance(self):
        """Test query performance with and without indexes."""
        database_path = self.db_manager.create_temp_database()
        client = SQLiteClient(database_path)
        
        client.execute_query("""
            CREATE TABLE query_test (
                id INTEGER PRIMARY KEY,
                category TEXT,
                value INTEGER
            )
        """)
        
        for i in range(500):
            category = f"Category {i % 10}"
            client.execute_query(
                "INSERT INTO query_test (category, value) VALUES (?, ?)",
                (category, i)
            )
        
        start_time = time.time()
        
        result = client.execute_query(
            "SELECT * FROM query_test WHERE category = ?",
            ("Category 5",)
        )
        rows_without_index = result.fetchall()
        
        end_time = time.time()
        time_without_index = end_time - start_time
        
        client.execute_query(
            "CREATE INDEX idx_category ON query_test(category)"
        )
        
        start_time = time.time()
        
        result = client.execute_query(
            "SELECT * FROM query_test WHERE category = ?",
            ("Category 5",)
        )
        rows_with_index = result.fetchall()
        
        end_time = time.time()
        time_with_index = end_time - start_time
        
        assert len(rows_without_index) == 50
        assert len(rows_with_index) == 50
        
        client.close_connection()


def test_basic_database_connection():
    """Basic database connection verification test."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_file:
        database_path = temp_file.name
    
    try:
        if SQLITE_CLIENT_AVAILABLE:
            client = SQLiteClient(database_path)
            assert client is not None
            if hasattr(client, 'is_connected'):
                connection_status = client.is_connected()
                assert connection_status is not False
            client.close_connection()
        else:
            connection = sqlite3.connect(database_path)
            assert connection is not None
            connection.close()
        
    finally:
        if os.path.exists(database_path):
            os.unlink(database_path)


def test_table_creation_and_basic_operations():
    """Test basic table creation and CRUD operations."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_file:
        database_path = temp_file.name
    
    try:
        if SQLITE_CLIENT_AVAILABLE:
            client = SQLiteClient(database_path)
            
            client.execute_query("""
                CREATE TABLE test_table (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    value INTEGER
                )
            """)
            
            client.execute_query(
                "INSERT INTO test_table (name, value) VALUES (?, ?)",
                ("Test Item", 42)
            )
            
            result = client.execute_query(
                "SELECT name, value FROM test_table WHERE name = ?",
                ("Test Item",)
            )
            row = result.fetchone()
            
            assert row is not None
            assert row[0] == "Test Item"
            assert row[1] == 42
            
            client.close_connection()
        else:
            connection = sqlite3.connect(database_path)
            connection.execute("""
                CREATE TABLE test_table (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    value INTEGER
                )
            """)
            
            connection.execute(
                "INSERT INTO test_table (name, value) VALUES (?, ?)",
                ("Test Item", 42)
            )
            
            cursor = connection.execute(
                "SELECT name, value FROM test_table WHERE name = ?",
                ("Test Item",)
            )
            row = cursor.fetchone()
            
            assert row is not None
            assert row[0] == "Test Item"
            assert row[1] == 42
            
            connection.close()
        
    finally:
        if os.path.exists(database_path):
            os.unlink(database_path)


def test_multiple_insert_and_select_operations():
    """Test multiple insert and select operations."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_file:
        database_path = temp_file.name
    
    try:
        if SQLITE_CLIENT_AVAILABLE:
            client = SQLiteClient(database_path)
        else:
            class SimpleClient:
                def __init__(self, db_path):
                    self.connection = sqlite3.connect(db_path)
                    self.connection.row_factory = sqlite3.Row
                
                def execute_query(self, query, params=None):
                    if params:
                        return self.connection.execute(query, params)
                    return self.connection.execute(query)
                
                def fetch_all(self, query, params=None):
                    cursor = self.execute_query(query, params)
                    return cursor.fetchall()
                
                def close_connection(self):
                    self.connection.close()
            
            client = SimpleClient(database_path)
        
        client.execute_query("""
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT,
                price REAL
            )
        """)
        
        products = [
            ("Product A", 10.99),
            ("Product B", 20.50),
            ("Product C", 15.75)
        ]
        
        for name, price in products:
            client.execute_query(
                "INSERT INTO products (name, price) VALUES (?, ?)",
                (name, price)
            )
        
        all_products = client.fetch_all("SELECT name, price FROM products ORDER BY name")
        
        assert len(all_products) == 3
        assert all_products[0][0] == "Product A"
        assert all_products[0][1] == 10.99
        assert all_products[1][0] == "Product B"
        assert all_products[1][1] == 20.50
        assert all_products[2][0] == "Product C"
        assert all_products[2][1] == 15.75
        
        client.close_connection()
        
    finally:
        if os.path.exists(database_path):
            os.unlink(database_path)


if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(0 if exit_code == 0 else 1)