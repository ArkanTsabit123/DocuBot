# DocuBot/src/ai_engine/prompt_templates.py

"""
Prompt Templates for DocuBot AI Engine.
Provides standardized prompt templates for various AI tasks.
"""

from typing import Dict, List, Any, Optional
from string import Template
import json


class PromptTemplates:
    """
    Manages prompt templates for different AI tasks.
    Supports template substitution and dynamic prompt generation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.templates = self._load_default_templates()
    
    def _load_default_templates(self) -> Dict[str, str]:
        """Load default prompt templates."""
        return {
            # Document-related prompts
            'document_summary': """
Please provide a summary of the following document.

DOCUMENT TITLE: ${document_title}
DOCUMENT CONTENT:
${document_content}

SUMMARY REQUIREMENTS:
- Extract key points and main ideas
- Identify important dates, names, and figures
- Highlight conclusions and recommendations
- Keep the summary concise but informative
- Use bullet points for clarity

Please provide a structured summary in the following format:
1. **Overview**: Brief description of the document
2. **Key Points**: Main ideas and findings
3. **Important Details**: Specific facts, figures, and names
4. **Conclusions**: Main conclusions or recommendations
""",
            
            'document_qa': """
Answer the following question based on the provided document context.

DOCUMENT CONTEXT:
${document_context}

QUESTION: ${question}

Please provide:
1. A direct answer to the question
2. Supporting evidence from the document
3. Context or explanation if needed
4. Reference to specific sections if available

If the answer cannot be found in the document, please state: 
"I cannot find the answer to this question in the provided document."
""",
            
            'document_chunk_summary': """
Summarize the following document chunk while maintaining context for the overall document.

DOCUMENT TITLE: ${document_title}
CURRENT CHUNK (Part ${chunk_number} of ${total_chunks}):
${chunk_content}

PREVIOUS CHUNK SUMMARY (if available):
${previous_summary}

Please provide a concise summary that:
1. Captures the main ideas of this chunk
2. Connects to the previous context if available
3. Highlights any new information
4. Prepares for the next chunk if relevant

Summary:
""",
            
            # RAG-specific prompts
            'rag_query': """
Based on the following context, please answer the user's question.

CONTEXT INFORMATION:
${context}

USER QUESTION: ${question}

INSTRUCTIONS:
1. Answer based ONLY on the provided context
2. If the context doesn't contain relevant information, say so
3. Cite specific parts of the context when possible
4. Provide a clear, concise answer
5. If multiple relevant pieces exist, synthesize them

ANSWER:
""",
            
            'rag_chat': """
You are DocuBot, an AI assistant specialized in document analysis and Q&A.

CONTEXT FROM DOCUMENTS:
${document_context}

CONVERSATION HISTORY:
${conversation_history}

USER INPUT: ${user_input}

Please respond by:
1. Using the document context when relevant
2. Acknowledging the conversation history
3. Providing accurate, helpful information
4. Being clear about limitations if information is not available

RESPONSE:
""",
            
            # Extraction prompts
            'extract_entities': """
Extract named entities from the following text.

TEXT:
${text}

Please identify and categorize:
1. Persons (names of people)
2. Organizations (companies, institutions)
3. Locations (places, addresses)
4. Dates and times
5. Key terms and concepts
6. Numerical values (amounts, statistics)

Format as JSON with categories as keys and lists of entities as values.
""",
            
            'extract_key_points': """
Extract the key points from the following text.

TEXT:
${text}

Please provide:
1. Main topic or subject
2. 3-5 key points or arguments
3. Important conclusions
4. Any action items or recommendations
5. Keywords that summarize the content

Format as a structured list with brief explanations.
""",
            
            # Analysis prompts
            'compare_documents': """
Compare and contrast the following two documents.

DOCUMENT 1: ${doc1_title}
CONTENT: ${doc1_summary}

DOCUMENT 2: ${doc2_title}
CONTENT: ${doc2_summary}

Please analyze:
1. Similarities in content, theme, or conclusions
2. Differences in perspective, data, or recommendations
3. Complementary information
4. Conflicting information (if any)
5. Overall relationship between the documents

Provide a structured comparison with specific examples.
""",
            
            'analyze_structure': """
Analyze the structure and organization of the following document.

DOCUMENT TITLE: ${document_title}
CONTENT OVERVIEW: ${content_overview}

Please identify:
1. Document type (report, article, manual, etc.)
2. Organizational structure (sections, headings, hierarchy)
3. Flow of information (logical progression)
4. Key structural elements (introduction, body, conclusion)
5. Any missing structural elements that would improve readability

Provide suggestions for better organization if applicable.
""",
            
            # Specialized prompts
            'legal_document_analysis': """
Analyze the following legal document.

DOCUMENT TYPE: ${document_type}
CONTENT: ${document_content}

Please identify:
1. Parties involved
2. Key obligations and responsibilities
3. Important dates and deadlines
4. Conditions and contingencies
5. Termination clauses
6. Legal terminology and definitions
7. Potential risks or concerns

Provide a plain English explanation of the document's purpose and implications.
""",
            
            'technical_document_qa': """
Answer technical questions about the following document.

DOCUMENT TYPE: ${document_type} (e.g., API documentation, technical manual, research paper)
CONTENT: ${document_content}

QUESTION: ${question}

Please provide:
1. Technical answer based on the documentation
2. Code examples if relevant
3. References to specific sections
4. Alternative approaches if mentioned
5. Limitations or caveats

If the documentation doesn't cover the question, suggest where to look for additional information.
""",
            
            'meeting_minutes_analysis': """
Analyze the following meeting minutes.

MEETING: ${meeting_title}
DATE: ${meeting_date}
PARTICIPANTS: ${participants}
CONTENT: ${minutes_content}

Please extract:
1. Decisions made
2. Action items with assignees and deadlines
3. Key discussion points
4. Open questions or unresolved issues
5. Next steps
6. Important announcements

Format as a structured summary for follow-up.
""",
            
            # System prompts
            'system_instruction': """
You are DocuBot, an AI assistant specialized in document analysis and knowledge management.

CAPABILITIES:
- Document summarization and analysis
- Question answering based on document content
- Entity extraction and information organization
- Document comparison and synthesis
- Technical documentation understanding

GUIDELINES:
1. Base responses on provided document context
2. Be accurate and cite sources when possible
3. Acknowledge limitations when information is insufficient
4. Provide clear, structured responses
5. Adapt tone to the document type (formal, technical, conversational)

USER REQUEST: ${user_request}
""",
            
            'error_recovery': """
The previous response encountered an error or was incomplete.

ERROR CONTEXT: ${error_context}
ORIGINAL REQUEST: ${original_request}
DOCUMENT CONTEXT: ${document_context}

Please provide an improved response that:
1. Addresses the original request more effectively
2. Uses the document context appropriately
3. Avoids the previous error pattern
4. Provides additional clarification if needed

IMPROVED RESPONSE:
""",
            
            'multi_document_synthesis': """
Synthesize information from multiple documents.

DOCUMENTS:
${documents_list}

USER QUESTION: ${question}

Please:
1. Identify relevant information from each document
2. Combine information into a coherent answer
3. Note conflicts or contradictions between documents
4. Provide source references for key points
5. Highlight gaps in the available information

SYNTHESIZED ANSWER:
"""
        }
    
    def get_template(self, template_name: str) -> Optional[str]:
        """Get a specific template by name."""
        return self.templates.get(template_name)
    
    def render_template(self, template_name: str, **kwargs) -> str:
        """
        Render a template with the provided variables.
        
        Args:
            template_name: Name of the template to render
            **kwargs: Variables to substitute in the template
            
        Returns:
            Rendered template string
        """
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        # Use string.Template for safe substitution
        t = Template(template)
        
        # Ensure all placeholders have values
        try:
            return t.substitute(**kwargs)
        except KeyError as e:
            missing_key = str(e).strip("'")
            raise ValueError(
                f"Missing required variable '{missing_key}' for template '{template_name}'"
            )
    
    def add_template(self, name: str, template: str) -> None:
        """Add a new template or update an existing one."""
        self.templates[name] = template
    
    def remove_template(self, name: str) -> bool:
        """Remove a template by name."""
        if name in self.templates:
            del self.templates[name]
            return True
        return False
    
    def list_templates(self) -> List[str]:
        """List all available template names."""
        return list(self.templates.keys())
    
    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """Get information about a template including required variables."""
        template = self.get_template(template_name)
        if not template:
            return {'exists': False}
        
        # Extract variable names from template
        import re
        variable_pattern = r'\${(\w+)}'
        variables = re.findall(variable_pattern, template)
        
        return {
            'exists': True,
            'name': template_name,
            'length': len(template),
            'required_variables': list(set(variables)),
            'variable_count': len(set(variables)),
            'preview': template[:200] + '...' if len(template) > 200 else template
        }
    
    def validate_template(self, template: str) -> tuple[bool, List[str]]:
        """
        Validate a template string.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if not template or not template.strip():
            issues.append("Template is empty")
            return False, issues
        
        # Check for proper formatting
        import re
        variable_pattern = r'\${(\w+)}'
        variables = re.findall(variable_pattern, template)
        
        # Check for unclosed variables
        if '${' in template and '}' not in template:
            issues.append("Unclosed variable placeholder")
        
        # Check variable naming
        for var in variables:
            if not var.replace('_', '').isalnum():
                issues.append(f"Invalid variable name: '{var}'")
        
        # Check template length
        if len(template) < 20:
            issues.append("Template is very short")
        elif len(template) > 10000:
            issues.append("Template is very long")
        
        return len(issues) == 0, issues
    
    def batch_render(self, template_name: str, variables_list: List[Dict[str, Any]]) -> List[str]:
        """Render the same template with multiple sets of variables."""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        results = []
        t = Template(template)
        
        for variables in variables_list:
            try:
                rendered = t.substitute(**variables)
                results.append(rendered)
            except KeyError as e:
                missing_key = str(e).strip("'")
                results.append(
                    f"ERROR: Missing variable '{missing_key}' for template '{template_name}'"
                )
        
        return results
    
    def save_templates(self, file_path: str) -> bool:
        """Save templates to a JSON file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.templates, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving templates: {e}")
            return False
    
    def load_templates(self, file_path: str) -> bool:
        """Load templates from a JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_templates = json.load(f)
            
            if isinstance(loaded_templates, dict):
                self.templates.update(loaded_templates)
                return True
            else:
                print("Error: Loaded templates are not a dictionary")
                return False
        except FileNotFoundError:
            print(f"Template file not found: {file_path}")
            return False
        except Exception as e:
            print(f"Error loading templates: {e}")
            return False
    
    def export_for_ui(self) -> List[Dict[str, Any]]:
        """Export templates in a format suitable for UI display."""
        ui_templates = []
        
        for name, template in self.templates.items():
            import re
            variables = list(set(re.findall(r'\${(\w+)}', template)))
            
            ui_templates.append({
                'name': name,
                'description': self._get_template_description(name),
                'category': self._get_template_category(name),
                'variables': variables,
                'preview': template[:150] + '...' if len(template) > 150 else template,
                'length': len(template)
            })
        
        return ui_templates
    
    def _get_template_description(self, template_name: str) -> str:
        """Get description for a template."""
        descriptions = {
            'document_summary': 'Summarize document content',
            'document_qa': 'Answer questions about documents',
            'document_chunk_summary': 'Summarize document chunks',
            'rag_query': 'RAG-based question answering',
            'rag_chat': 'Document-aware chat responses',
            'extract_entities': 'Extract named entities from text',
            'extract_key_points': 'Extract key points from text',
            'compare_documents': 'Compare multiple documents',
            'analyze_structure': 'Analyze document structure',
            'legal_document_analysis': 'Analyze legal documents',
            'technical_document_qa': 'Technical documentation Q&A',
            'meeting_minutes_analysis': 'Analyze meeting minutes',
            'system_instruction': 'System instructions for AI',
            'error_recovery': 'Error recovery and improvement',
            'multi_document_synthesis': 'Synthesize multiple documents'
        }
        return descriptions.get(template_name, 'General purpose template')
    
    def _get_template_category(self, template_name: str) -> str:
        """Categorize templates."""
        categories = {
            'document_summary': 'Document Processing',
            'document_qa': 'Document Processing',
            'document_chunk_summary': 'Document Processing',
            'rag_query': 'RAG System',
            'rag_chat': 'RAG System',
            'extract_entities': 'Information Extraction',
            'extract_key_points': 'Information Extraction',
            'compare_documents': 'Document Analysis',
            'analyze_structure': 'Document Analysis',
            'legal_document_analysis': 'Specialized Analysis',
            'technical_document_qa': 'Specialized Analysis',
            'meeting_minutes_analysis': 'Specialized Analysis',
            'system_instruction': 'System Configuration',
            'error_recovery': 'System Configuration',
            'multi_document_synthesis': 'Document Analysis'
        }
        return categories.get(template_name, 'General')


# Factory function
def get_prompt_templates(config: Optional[Dict[str, Any]] = None) -> PromptTemplates:
    """
    Get or create a PromptTemplates instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        PromptTemplates instance
    """
    return PromptTemplates(config)


# Test function
def test_prompt_templates():
    """Test the prompt templates functionality."""
    print("Testing PromptTemplates...")
    print("=" * 60)
    
    templates = PromptTemplates()
    
    # Test basic functionality
    print(f"Available templates: {len(templates.list_templates())}")
    print(f"Template names: {templates.list_templates()[:5]}...")
    
    # Test template rendering
    try:
        rendered = templates.render_template(
            'document_summary',
            document_title="Test Document",
            document_content="This is a test document content."
        )
        print(f"\n✓ Template rendering successful")
        print(f"  Rendered length: {len(rendered)} characters")
        print(f"  Preview: {rendered[:100]}...")
    except Exception as e:
        print(f"\n✗ Template rendering failed: {e}")
    
    # Test template info
    template_info = templates.get_template_info('document_qa')
    print(f"\n✓ Template info for 'document_qa':")
    print(f"  Variables required: {template_info.get('required_variables', [])}")
    print(f"  Variable count: {template_info.get('variable_count', 0)}")
    
    # Test validation
    valid, issues = templates.validate_template("Test ${variable} template")
    if valid:
        print(f"\n✓ Template validation successful")
    else:
        print(f"\n✗ Template validation failed: {issues}")
    
    print("\n" + "=" * 60)
    print("PromptTemplates test complete")


if __name__ == "__main__":
    test_prompt_templates()