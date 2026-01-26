"""
Document Tagging Module for DocuBot
Provides automatic tagging and categorization for documents.
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DocumentTag:
    """Represents a tag for document classification."""
    
    def __init__(self, name: str, category: str, confidence: float = 1.0):
        self.name = name
        self.category = category
        self.confidence = confidence
        self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'category': self.category,
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat()
        }
    
    def __repr__(self):
        return f"DocumentTag(name='{self.name}', category='{self.category}', confidence={self.confidence})"


class TagCategory:
    """Represents a category of tags."""
    
    def __init__(self, name: str, description: str, priority: int = 0):
        self.name = name
        self.description = description
        self.priority = priority
        self.tags: List[DocumentTag] = []
    
    def add_tag(self, tag: DocumentTag):
        """Add a tag to this category."""
        self.tags.append(tag)
    
    def get_tags(self) -> List[DocumentTag]:
        """Get all tags in this category."""
        return self.tags
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'priority': self.priority,
            'tag_count': len(self.tags)
        }


class TagRule:
    """Rule for automatic tagging based on content patterns."""
    
    def __init__(self, name: str, pattern: str, tag_name: str, 
                 category: str, priority: int = 0):
        self.name = name
        self.pattern = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        self.tag_name = tag_name
        self.category = category
        self.priority = priority
    
    def match(self, text: str) -> Optional[Tuple[str, float]]:
        """Check if text matches this rule."""
        matches = self.pattern.findall(text)
        if matches:
            # Confidence based on number of matches
            confidence = min(0.5 + (len(matches) * 0.1), 1.0)
            return self.tag_name, confidence
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'pattern': self.pattern.pattern,
            'tag_name': self.tag_name,
            'category': self.category,
            'priority': self.priority
        }


class TaggingRuleSet:
    """Collection of tagging rules."""
    
    def __init__(self):
        self.rules: List[TagRule] = []
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize with default tagging rules."""
        default_rules = [
            # Document type rules
            TagRule("contract", r"(contract|agreement|terms and conditions)", 
                   "contract", "document_type", 10),
            TagRule("invoice", r"(invoice|receipt|bill|payment)", 
                   "invoice", "document_type", 10),
            TagRule("report", r"(report|analysis|findings|conclusion)", 
                   "report", "document_type", 8),
            TagRule("proposal", r"(proposal|offer|quotation|bid)", 
                   "proposal", "document_type", 9),
            TagRule("manual", r"(manual|guide|instruction|tutorial)", 
                   "manual", "document_type", 7),
            TagRule("meeting", r"(meeting|minutes|agenda|notes)", 
                   "meeting_notes", "document_type", 8),
            
            # Technology rules
            TagRule("python", r"(python|django|flask|numpy|pandas)", 
                   "python", "technology", 9),
            TagRule("javascript", r"(javascript|node\.?js|react|vue|angular)", 
                   "javascript", "technology", 9),
            TagRule("ai", r"(artificial intelligence|machine learning|deep learning|neural network|llm)", 
                   "ai_ml", "technology", 10),
            TagRule("database", r"(database|sql|mysql|postgresql|mongodb)", 
                   "database", "technology", 8),
            TagRule("cloud", r"(cloud|aws|azure|gcp|docker|kubernetes)", 
                   "cloud", "technology", 9),
            
            # Business rules
            TagRule("finance", r"(finance|financial|budget|revenue|profit)", 
                   "finance", "business", 9),
            TagRule("legal", r"(legal|law|regulation|compliance|clause)", 
                   "legal", "business", 10),
            TagRule("hr", r"(human resources|hr|employee|recruitment|hiring)", 
                   "hr", "business", 8),
            TagRule("marketing", r"(marketing|advertising|campaign|brand)", 
                   "marketing", "business", 7),
            TagRule("project", r"(project|milestone|deadline|deliverable)", 
                   "project", "business", 8),
            
            # Content rules
            TagRule("confidential", r"(confidential|secret|proprietary|classified)", 
                   "confidential", "content_type", 10),
            TagRule("draft", r"(draft|preliminary|temporary|working copy)", 
                   "draft", "content_type", 6),
            TagRule("final", r"(final|approved|completed|signed)", 
                   "final", "content_type", 7),
            TagRule("urgent", r"(urgent|asap|immediate|priority)", 
                   "urgent", "content_type", 9),
        ]
        
        self.rules.extend(default_rules)
        logger.info(f"Initialized tagging rule set with {len(default_rules)} rules")
    
    def add_rule(self, rule: TagRule):
        """Add a custom rule."""
        self.rules.append(rule)
        self.rules.sort(key=lambda x: x.priority, reverse=True)
        logger.debug(f"Added tagging rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a rule by name."""
        initial_count = len(self.rules)
        self.rules = [r for r in self.rules if r.name != rule_name]
        
        if len(self.rules) < initial_count:
            logger.debug(f"Removed tagging rule: {rule_name}")
            return True
        return False
    
    def apply_rules(self, text: str) -> List[DocumentTag]:
        """Apply all rules to text and return matching tags."""
        tags = []
        seen_tags = set()
        
        for rule in self.rules:
            result = rule.match(text)
            if result:
                tag_name, confidence = result
                tag_key = f"{rule.category}:{tag_name}"
                
                # Avoid duplicate tags
                if tag_key not in seen_tags:
                    tag = DocumentTag(tag_name, rule.category, confidence)
                    tags.append(tag)
                    seen_tags.add(tag_key)
                    
                    if len(tags) >= 10:  # Limit number of tags
                        break
        
        # Sort by confidence (highest first)
        tags.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.debug(f"Applied tagging rules, found {len(tags)} tags")
        return tags
    
    def get_rules_by_category(self, category: str) -> List[TagRule]:
        """Get all rules in a specific category."""
        return [r for r in self.rules if r.category == category]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'rule_count': len(self.rules),
            'rules': [rule.to_dict() for rule in self.rules],
            'categories': list(set(rule.category for rule in self.rules))
        }


class Tagger:
    """
    Main tagging engine for DocuBot.
    Provides automatic document tagging and categorization.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the tagger.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.rule_set = TaggingRuleSet()
        self.custom_rules_loaded = False
        
        # Load custom rules if specified
        self._load_custom_rules()
        
        # Statistics
        self.stats = {
            'documents_tagged': 0,
            'tags_applied': 0,
            'rules_executed': 0,
            'last_tagged': None
        }
        
        logger.info("Tagger initialized")
    
    def _load_custom_rules(self):
        """Load custom tagging rules from configuration."""
        if not self.config:
            return
        
        custom_rules = self.config.get('tagging', {}).get('custom_rules', [])
        
        for rule_config in custom_rules:
            try:
                rule = TagRule(
                    name=rule_config.get('name', 'unnamed_rule'),
                    pattern=rule_config['pattern'],
                    tag_name=rule_config['tag_name'],
                    category=rule_config.get('category', 'custom'),
                    priority=rule_config.get('priority', 5)
                )
                self.rule_set.add_rule(rule)
                self.custom_rules_loaded = True
            except KeyError as e:
                logger.warning(f"Invalid custom rule configuration: missing {e}")
            except Exception as e:
                logger.warning(f"Failed to load custom rule: {e}")
        
        if self.custom_rules_loaded:
            logger.info(f"Loaded {len(custom_rules)} custom tagging rules")
    
    def tag_document(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[DocumentTag]:
        """
        Tag a document based on its content.
        
        Args:
            text: Document text content
            metadata: Optional document metadata
            
        Returns:
            List of DocumentTag objects
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for tagging")
            return []
        
        # Apply rules to extract tags
        tags = self.rule_set.apply_rules(text)
        
        # Add metadata-based tags if available
        if metadata:
            metadata_tags = self._extract_tags_from_metadata(metadata)
            tags.extend(metadata_tags)
        
        # Remove duplicates (same name and category)
        unique_tags = []
        seen = set()
        
        for tag in tags:
            identifier = f"{tag.category}:{tag.name}"
            if identifier not in seen:
                unique_tags.append(tag)
                seen.add(identifier)
        
        # Update statistics
        self.stats['documents_tagged'] += 1
        self.stats['tags_applied'] += len(unique_tags)
        self.stats['rules_executed'] += len(self.rule_set.rules)
        self.stats['last_tagged'] = datetime.now().isoformat()
        
        logger.info(f"Tagged document with {len(unique_tags)} tags")
        return unique_tags
    
    def _extract_tags_from_metadata(self, metadata: Dict[str, Any]) -> List[DocumentTag]:
        """Extract tags from document metadata."""
        tags = []
        
        # Extract from file extension
        if 'file_extension' in metadata:
            ext = metadata['file_extension'].lower()
            if ext in ['.pdf', '.docx', '.txt', '.md']:
                tags.append(DocumentTag(ext[1:], "file_format", 0.9))
        
        # Extract from file name patterns
        if 'file_name' in metadata:
            filename = metadata['file_name'].lower()
            
            # Check for common patterns
            patterns = [
                ('report', r'report'),
                ('invoice', r'invoice'),
                ('contract', r'contract'),
                ('proposal', r'proposal'),
                ('manual', r'manual|guide'),
                ('draft', r'draft'),
                ('final', r'final'),
            ]
            
            for tag_name, pattern in patterns:
                if re.search(pattern, filename):
                    tags.append(DocumentTag(tag_name, "file_name_pattern", 0.8))
        
        # Extract from creation/modification dates
        if 'created_date' in metadata:
            try:
                created_date = datetime.fromisoformat(metadata['created_date'].replace('Z', '+00:00'))
                current_year = datetime.now().year
                document_year = created_date.year
                
                if document_year == current_year:
                    tags.append(DocumentTag("current_year", "date", 0.7))
                elif document_year < current_year - 1:
                    tags.append(DocumentTag("archival", "date", 0.6))
            except:
                pass
        
        return tags
    
    def tag_document_from_file(self, file_path: Path) -> List[DocumentTag]:
        """
        Tag a document from a file path.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of DocumentTag objects
        """
        try:
            # For now, we'll just extract basic metadata
            # In a real implementation, you would extract text content
            metadata = {
                'file_name': file_path.name,
                'file_extension': file_path.suffix,
                'file_size': file_path.stat().st_size,
                'created_date': datetime.fromtimestamp(file_path.stat().st_ctime).isoformat()
            }
            
            # Create mock text based on file name for demonstration
            mock_text = f"Document: {file_path.name}"
            
            return self.tag_document(mock_text, metadata)
            
        except Exception as e:
            logger.error(f"Failed to tag document from file {file_path}: {e}")
            return []
    
    def add_custom_rule(self, name: str, pattern: str, tag_name: str, 
                       category: str = "custom", priority: int = 5) -> bool:
        """
        Add a custom tagging rule.
        
        Args:
            name: Rule name
            pattern: Regex pattern
            tag_name: Tag to apply on match
            category: Tag category
            priority: Rule priority
            
        Returns:
            True if rule was added successfully
        """
        try:
            rule = TagRule(name, pattern, tag_name, category, priority)
            self.rule_set.add_rule(rule)
            
            # Save to config if config is available
            if self.config:
                if 'tagging' not in self.config:
                    self.config['tagging'] = {}
                if 'custom_rules' not in self.config['tagging']:
                    self.config['tagging']['custom_rules'] = []
                
                rule_config = {
                    'name': name,
                    'pattern': pattern,
                    'tag_name': tag_name,
                    'category': category,
                    'priority': priority
                }
                
                # Remove existing rule with same name
                self.config['tagging']['custom_rules'] = [
                    r for r in self.config['tagging']['custom_rules'] 
                    if r.get('name') != name
                ]
                
                self.config['tagging']['custom_rules'].append(rule_config)
            
            logger.info(f"Added custom rule: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add custom rule {name}: {e}")
            return False
    
    def remove_custom_rule(self, rule_name: str) -> bool:
        """Remove a custom rule by name."""
        success = self.rule_set.remove_rule(rule_name)
        
        # Also remove from config
        if success and self.config and 'tagging' in self.config:
            if 'custom_rules' in self.config['tagging']:
                self.config['tagging']['custom_rules'] = [
                    r for r in self.config['tagging']['custom_rules']
                    if r.get('name') != rule_name
                ]
        
        return success
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tagging statistics."""
        return {
            **self.stats,
            'rule_count': len(self.rule_set.rules),
            'custom_rules_loaded': self.custom_rules_loaded,
            'rule_categories': list(set(rule.category for rule in self.rule_set.rules))
        }
    
    def export_rules(self, file_path: Path) -> bool:
        """Export tagging rules to JSON file."""
        try:
            rules_data = self.rule_set.to_dict()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(rules_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported {rules_data['rule_count']} rules to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export rules: {e}")
            return False
    
    def import_rules(self, file_path: Path) -> bool:
        """Import tagging rules from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                rules_data = json.load(f)
            
            imported_count = 0
            for rule_data in rules_data.get('rules', []):
                try:
                    rule = TagRule(
                        name=rule_data['name'],
                        pattern=rule_data['pattern'],
                        tag_name=rule_data['tag_name'],
                        category=rule_data.get('category', 'imported'),
                        priority=rule_data.get('priority', 5)
                    )
                    self.rule_set.add_rule(rule)
                    imported_count += 1
                except KeyError as e:
                    logger.warning(f"Invalid rule data in import: missing {e}")
                except Exception as e:
                    logger.warning(f"Failed to import rule: {e}")
            
            logger.info(f"Imported {imported_count} rules from {file_path}")
            return imported_count > 0
            
        except Exception as e:
            logger.error(f"Failed to import rules: {e}")
            return False
    
    def suggest_tags(self, text: str, max_suggestions: int = 5) -> List[Dict[str, Any]]:
        """
        Suggest tags for text without applying them.
        
        Args:
            text: Text to analyze
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of tag suggestions with confidence scores
        """
        tags = self.tag_document(text)
        
        suggestions = []
        for tag in tags[:max_suggestions]:
            suggestions.append({
                'name': tag.name,
                'category': tag.category,
                'confidence': tag.confidence,
                'description': f"Based on pattern matching in '{tag.category}' category"
            })
        
        return suggestions
    
    def validate_rule(self, pattern: str, test_text: str) -> Dict[str, Any]:
        """
        Validate a regex pattern and test it against sample text.
        
        Args:
            pattern: Regex pattern to validate
            test_text: Text to test against
            
        Returns:
            Validation results
        """
        try:
            compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            matches = compiled.findall(test_text)
            
            return {
                'valid': True,
                'match_count': len(matches),
                'matches': matches[:5],  # Limit matches for display
                'message': f"Pattern is valid, found {len(matches)} matches"
            }
            
        except re.error as e:
            return {
                'valid': False,
                'match_count': 0,
                'matches': [],
                'error': str(e),
                'message': f"Invalid regex pattern: {str(e)}"
            }
        except Exception as e:
            return {
                'valid': False,
                'match_count': 0,
                'matches': [],
                'error': str(e),
                'message': f"Error validating pattern: {str(e)}"
            }


def get_tagger(config: Optional[Dict[str, Any]] = None) -> Tagger:
    """
    Factory function to get a Tagger instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tagger instance
    """
    return Tagger(config)


if __name__ == "__main__":
    """
    Command-line test for the tagger.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Document Tagger")
    parser.add_argument("--text", type=str, help="Text to tag")
    parser.add_argument("--file", type=str, help="File to tag")
    parser.add_argument("--test-rule", type=str, help="Test a regex pattern")
    parser.add_argument("--export-rules", type=str, help="Export rules to file")
    parser.add_argument("--import-rules", type=str, help="Import rules from file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)-8s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    print("\n" + "=" * 80)
    print("DOCUBOT DOCUMENT TAGGER - TEST")
    print("=" * 80)
    
    try:
        # Create tagger
        tagger = Tagger()
        
        if args.export_rules:
            print(f"\nExporting rules to: {args.export_rules}")
            success = tagger.export_rules(Path(args.export_rules))
            if success:
                print("✓ Rules exported successfully")
            else:
                print("✗ Failed to export rules")
        
        elif args.import_rules:
            print(f"\nImporting rules from: {args.import_rules}")
            success = tagger.import_rules(Path(args.import_rules))
            if success:
                print("✓ Rules imported successfully")
            else:
                print("✗ Failed to import rules")
        
        elif args.test_rule:
            print(f"\nTesting rule pattern: {args.test_rule}")
            test_text = "This is a test document about Python and machine learning."
            result = tagger.validate_rule(args.test_rule, test_text)
            
            print(f"Valid: {result['valid']}")
            print(f"Matches: {result['match_count']}")
            if result['valid'] and result['matches']:
                print(f"Sample matches: {result['matches']}")
            elif not result['valid']:
                print(f"Error: {result.get('error', 'Unknown error')}")
        
        elif args.file:
            print(f"\nTagging file: {args.file}")
            file_path = Path(args.file)
            
            if file_path.exists():
                tags = tagger.tag_document_from_file(file_path)
                print(f"Found {len(tags)} tags:")
                for tag in tags:
                    print(f"  • {tag.name} ({tag.category}) - Confidence: {tag.confidence:.2f}")
            else:
                print(f"✗ File not found: {args.file}")
        
        elif args.text:
            print(f"\nTagging text: {args.text[:100]}...")
            tags = tagger.tag_document(args.text)
            
            print(f"Found {len(tags)} tags:")
            for tag in tags:
                print(f"  • {tag.name} ({tag.category}) - Confidence: {tag.confidence:.2f}")
            
            # Test suggestions
            suggestions = tagger.suggest_tags(args.text, max_suggestions=3)
            print(f"\nTop suggestions:")
            for suggestion in suggestions:
                print(f"  • {suggestion['name']} - Confidence: {suggestion['confidence']:.2f}")
        
        else:
            # Run default tests
            print("\nRunning default tests...")
            
            test_cases = [
                ("This is a Python programming guide with machine learning examples.", "AI/Programming"),
                ("The financial report shows increased revenue and profit margins.", "Finance"),
                ("Meeting minutes from the project planning session.", "Business"),
                ("Legal contract with confidential terms and conditions.", "Legal/Confidential"),
            ]
            
            for test_text, expected in test_cases:
                print(f"\nTest: {expected}")
                print(f"Text: {test_text[:60]}...")
                
                tags = tagger.tag_document(test_text)
                print(f"Tags found: {len(tags)}")
                
                for tag in tags:
                    print(f"  • {tag.name} ({tag.category}) - {tag.confidence:.2f}")
        
        # Show statistics
        stats = tagger.get_statistics()
        print(f"\nStatistics:")
        print(f"  Documents tagged: {stats['documents_tagged']}")
        print(f"  Total tags applied: {stats['tags_applied']}")
        print(f"  Rules available: {stats['rule_count']}")
        print(f"  Categories: {', '.join(stats['rule_categories'])}")
        
        print("\n" + "=" * 80)
        print("TEST COMPLETE")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nERROR: Test failed: {e}")
        import traceback
        traceback.print_exc()