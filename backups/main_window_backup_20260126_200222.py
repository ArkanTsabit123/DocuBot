# docubot/src/ui/desktop/main_window.py

"""
main_window.py - DocuBot Main Desktop Window Module

Primary application window implementing the complete desktop interface
for DocuBot using CustomTkinter with modern UI/UX patterns.
"""

import os
import sys
import json
import tkinter as tk
import customtkinter as ctk
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import threading
import queue

current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "..", "..", "..", ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.core.config import AppConfig
    from src.core.app import DocuBotCore
    from src.document_processing.processor import DocumentProcessor
    from src.ai_engine.llm_client import LLMClient
    from src.database.sqlite_client import DatabaseClient
    from src.utilities.logger import get_logger
    
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Import warning: {e}")
    print("Using mock classes for development environment")
    IMPORT_SUCCESS = False
    
    class AppConfig:
        def __init__(self):
            self.app = type('obj', (object,), {
                'version': '1.0.0',
                'name': 'DocuBot'
            })
            self.ui = type('obj', (object,), {
                'theme': 'dark',
                'font_size': 12,
                'language': 'en',
                'enable_animations': True,
                'auto_save_interval': 60
            })
            self.ai = type('obj', (object,), {
                'llm': type('obj', (object,), {
                    'model': 'llama2:7b',
                    'temperature': 0.1,
                    'max_tokens': 1024,
                    'context_window': 4096
                }),
                'rag': type('obj', (object,), {
                    'top_k': 5,
                    'similarity_threshold': 0.7,
                    'enable_hybrid_search': True
                }),
                'embeddings': type('obj', (object,), {
                    'model': 'all-MiniLM-L6-v2',
                    'dimensions': 384,
                    'device': 'cpu'
                })
            })
            self.storage = type('obj', (object,), {
                'backup_enabled': False,
                'max_documents': 10000,
                'auto_cleanup_days': 90
            })
            self.performance = type('obj', (object,), {
                'max_workers': 4,
                'cache_enabled': True,
                'cache_size_mb': 500
            })
            self.privacy = type('obj', (object,), {
                'telemetry': False,
                'auto_update_check': False,
                'crash_reports': False
            })
    
    class DocuBotCore:
        def __init__(self, config):
            self.config = config
            print("Mock DocuBotCore initialized")
        
        def process_query(self, query):
            return {
                'answer': f"This is a mock response to: '{query}'\n\nDocuBot is running in development mode with mock components.",
                'sources': ['Mock Document 1.pdf', 'Mock Document 2.txt']
            }
    
    class DocumentProcessor:
        def __init__(self, config):
            self.config = config
            print("Mock DocumentProcessor initialized")
        
        def process_document(self, file_path):
            import time
            time.sleep(1)
            return {
                'status': 'success',
                'document_id': os.path.basename(file_path),
                'chunks_processed': 5,
                'processing_time': 1.0,
                'error': None
            }
    
    class LLMClient:
        def __init__(self, config):
            self.config = config
            print("Mock LLMClient initialized")
        
        def set_model(self, model):
            print(f"Mock: Model changed to {model}")
        
        def set_temperature(self, temp):
            print(f"Mock: Temperature changed to {temp}")
    
    class DatabaseClient:
        def __init__(self, config):
            self.config = config
            print("Mock DatabaseClient initialized")
            self.documents = []
        
        def get_all_documents(self):
            return self.documents
        
        def get_document_count(self):
            return len(self.documents)
        
        def delete_document(self, doc_id):
            self.documents = [d for d in self.documents if d.get('id') != doc_id]
        
        def create_conversation(self, title):
            return f"conv-{datetime.now().timestamp()}"
        
        def get_document_chunks(self, doc_id):
            return [
                {'text_content': 'This is a mock chunk from the document.'},
                {'text_content': 'Document processing is simulated in development mode.'}
            ]
    
    def get_logger(name):
        import logging
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


class MainWindow(ctk.CTk):
    def __init__(self, config: Optional[AppConfig] = None):
        super().__init__()
        
        self.logger = get_logger(__name__)
        self.app_config = config or AppConfig()
        self.core_app = None
        self.document_processor = None
        self.llm_client = None
        self.db_client = None
        
        self.current_conversation_id = None
        self.active_documents = []
        self.processing_queue = queue.Queue()
        self.is_processing = False
        
        self.setup_window_configuration()
        self.initialize_application_components()
        self.create_ui_structure()
        self.setup_event_bindings()
        self.apply_initial_settings()
        
        self.logger.info("Main window initialized successfully")
    
    def setup_window_configuration(self) -> None:
        self.title(f"DocuBot - Personal AI Knowledge Assistant v{self.app_config.app.version}")
        
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        
        window_width = min(1400, int(screen_width * 0.9))
        window_height = min(800, int(screen_height * 0.85))
        
        x_position = (screen_width - window_width) // 2
        y_position = (screen_height - window_height) // 2
        
        self.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
        self.minsize(1000, 600)
        
        ctk.set_appearance_mode(self.app_config.ui.theme)
        ctk.set_default_color_theme("blue")
        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.configure_grid_structure()
    
    def configure_grid_structure(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)
    
    def initialize_application_components(self) -> None:
        self.show_loading_screen("Initializing DocuBot...")
        
        def initialize_components():
            try:
                if IMPORT_SUCCESS:
                    self.core_app = DocuBotCore(self.app_config)
                    self.document_processor = DocumentProcessor(self.app_config)
                    self.llm_client = LLMClient(self.app_config)
                    self.db_client = DatabaseClient(self.app_config)
                else:
                    self.logger.warning("Using mock components for development")
                    self.core_app = DocuBotCore(self.app_config)
                    self.document_processor = DocumentProcessor(self.app_config)
                    self.llm_client = LLMClient(self.app_config)
                    self.db_client = DatabaseClient(self.app_config)
                
                self.after(0, self.hide_loading_screen)
                self.after(0, self.update_status_bar, "Ready - DocuBot Initialized")
                self.logger.info("Application components initialized successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize components: {e}")
                self.after(0, self.show_error_dialog, 
                          "Initialization Error", 
                          f"Failed to initialize application: {str(e)}")
                self.after(0, self.hide_loading_screen)
        
        init_thread = threading.Thread(target=initialize_components, daemon=True)
        init_thread.start()
    
    def show_loading_screen(self, message: str = "Loading...") -> None:
        self.loading_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.loading_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        loading_label = ctk.CTkLabel(
            self.loading_frame,
            text=message,
            font=ctk.CTkFont(size=16, weight="bold")
        )
        loading_label.pack(pady=20)
        
        self.progress_bar = ctk.CTkProgressBar(self.loading_frame, width=300)
        self.progress_bar.pack(pady=10)
        self.progress_bar.start()
        
        self.update()
    
    def hide_loading_screen(self) -> None:
        if hasattr(self, 'loading_frame'):
            self.loading_frame.destroy()
            del self.loading_frame
    
    def create_ui_structure(self) -> None:
        self.create_menu_bar()
        self.create_main_container()
        self.create_status_bar()
        self.create_document_panel()
        self.create_chat_panel()
        self.create_settings_panel()
    
    def create_menu_bar(self) -> None:
        menu_bar = tk.Menu(self)
        super().config(menu=menu_bar)
        
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="New Conversation", 
                            command=self.new_conversation,
                            accelerator="Ctrl+N")
        file_menu.add_command(label="Open Document...", 
                            command=self.open_document_dialog,
                            accelerator="Ctrl+O")
        file_menu.add_command(label="Open Folder...", 
                            command=self.open_folder_dialog,
                            accelerator="Ctrl+Shift+O")
        file_menu.add_separator()
        file_menu.add_command(label="Save Conversation...", 
                            command=self.save_conversation,
                            accelerator="Ctrl+S")
        file_menu.add_command(label="Export Conversation...", 
                            command=self.export_conversation)
        file_menu.add_separator()
        file_menu.add_command(label="Settings", 
                            command=self.open_settings_dialog,
                            accelerator="Ctrl+,")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", 
                            command=self.on_closing,
                            accelerator="Ctrl+Q")
        menu_bar.add_cascade(label="File", menu=file_menu)
        
        edit_menu = tk.Menu(menu_bar, tearoff=0)
        edit_menu.add_command(label="Preferences", 
                            command=self.open_preferences_dialog)
        edit_menu.add_command(label="Keyboard Shortcuts", 
                            command=self.show_keyboard_shortcuts)
        menu_bar.add_cascade(label="Edit", menu=edit_menu)
        
        view_menu = tk.Menu(menu_bar, tearoff=0)
        view_menu.add_command(label="Toggle Theme", 
                            command=self.toggle_theme,
                            accelerator="Ctrl+T")
        view_menu.add_command(label="Increase Font Size", 
                            command=self.increase_font_size,
                            accelerator="Ctrl++")
        view_menu.add_command(label="Decrease Font Size", 
                            command=self.decrease_font_size,
                            accelerator="Ctrl+-")
        view_menu.add_separator()
        view_menu.add_command(label="Reset Layout", 
                            command=self.reset_ui_layout)
        menu_bar.add_cascade(label="View", menu=view_menu)
        
        tools_menu = tk.Menu(menu_bar, tearoff=0)
        tools_menu.add_command(label="Document Management", 
                            command=self.open_document_manager)
        tools_menu.add_command(label="Vector Database", 
                            command=self.open_vector_db_manager)
        tools_menu.add_command(label="Model Management", 
                            command=self.open_model_manager)
        menu_bar.add_cascade(label="Tools", menu=tools_menu)
        
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="Documentation", 
                            command=self.open_documentation,
                            accelerator="F1")
        help_menu.add_command(label="Keyboard Shortcuts", 
                            command=self.show_keyboard_shortcuts)
        help_menu.add_separator()
        help_menu.add_command(label="Check for Updates", 
                            command=self.check_for_updates)
        help_menu.add_command(label="About DocuBot", 
                            command=self.show_about_dialog)
        menu_bar.add_cascade(label="Help", menu=help_menu)
    
    def create_main_container(self) -> None:
        self.main_container = ctk.CTkFrame(self)
        self.main_container.grid(row=0, column=0, columnspan=3, 
                                padx=10, pady=(0, 10), sticky="nsew")
        
        self.main_container.grid_columnconfigure(0, weight=1, minsize=250)
        self.main_container.grid_columnconfigure(1, weight=3, minsize=600)
        self.main_container.grid_columnconfigure(2, weight=1, minsize=250)
        self.main_container.grid_rowconfigure(0, weight=1)
    
    def create_document_panel(self) -> None:
        self.document_panel = ctk.CTkFrame(self.main_container)
        self.document_panel.grid(row=0, column=0, padx=(0, 5), pady=5, sticky="nsew")
        
        panel_header = ctk.CTkFrame(self.document_panel, height=40)
        panel_header.pack(fill=tk.X, padx=5, pady=(5, 0))
        
        title_label = ctk.CTkLabel(
            panel_header,
            text="Documents",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        title_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.document_count_label = ctk.CTkLabel(
            panel_header,
            text="(0)",
            font=ctk.CTkFont(size=12)
        )
        self.document_count_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.documents_listbox = tk.Listbox(
            self.document_panel,
            bg=self.get_listbox_background_color(),
            fg=self.get_listbox_foreground_color(),
            selectbackground="#3b8ed0",
            selectforeground="white",
            font=("Segoe UI", 11),
            relief=tk.FLAT,
            highlightthickness=0
        )
        
        listbox_scrollbar = ctk.CTkScrollbar(self.document_panel)
        listbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 5), pady=5)
        
        self.documents_listbox.config(yscrollcommand=listbox_scrollbar.set)
        listbox_scrollbar.configure(command=self.documents_listbox.yview)
        self.documents_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        button_frame = ctk.CTkFrame(self.document_panel)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        button_frame.grid_columnconfigure(2, weight=1)
        
        self.upload_btn = ctk.CTkButton(
            button_frame,
            text="Upload",
            command=self.upload_document,
            width=80
        )
        self.upload_btn.grid(row=0, column=0, padx=2, pady=2)
        
        self.remove_btn = ctk.CTkButton(
            button_frame,
            text="Remove",
            command=self.remove_selected_document,
            width=80,
            fg_color="#d9534f",
            hover_color="#c9302c"
        )
        self.remove_btn.grid(row=0, column=1, padx=2, pady=2)
        
        self.refresh_btn = ctk.CTkButton(
            button_frame,
            text="Refresh",
            command=self.refresh_document_list,
            width=80
        )
        self.refresh_btn.grid(row=0, column=2, padx=2, pady=2)
        
        self.documents_listbox.bind("<<ListboxSelect>>", self.on_document_selected)
        self.documents_listbox.bind("<Double-Button-1>", self.on_document_double_click)
    
    def get_listbox_background_color(self) -> str:
        if self.app_config.ui.theme == "dark":
            return "#2b2b2b"
        return "#f0f0f0"
    
    def get_listbox_foreground_color(self) -> str:
        if self.app_config.ui.theme == "dark":
            return "white"
        return "black"
    
    def create_chat_panel(self) -> None:
        self.chat_panel = ctk.CTkFrame(self.main_container)
        self.chat_panel.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        self.chat_panel.grid_columnconfigure(0, weight=1)
        self.chat_panel.grid_rowconfigure(0, weight=1)
        self.chat_panel.grid_rowconfigure(1, weight=0)
        
        self.chat_text = ctk.CTkTextbox(
            self.chat_panel,
            font=ctk.CTkFont(size=self.app_config.ui.font_size),
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.chat_text.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        input_frame = ctk.CTkFrame(self.chat_panel)
        input_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
        input_frame.grid_columnconfigure(0, weight=1)
        input_frame.grid_columnconfigure(1, weight=0)
        
        self.query_entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="Ask a question about your documents...",
            font=ctk.CTkFont(size=self.app_config.ui.font_size)
        )
        self.query_entry.grid(row=0, column=0, padx=(0, 5), pady=5, sticky="ew")
        self.query_entry.bind("<Return>", lambda e: self.process_user_query())
        
        self.send_btn = ctk.CTkButton(
            input_frame,
            text="Send",
            command=self.process_user_query,
            width=80
        )
        self.send_btn.grid(row=0, column=1, padx=(5, 0), pady=5)
        
        self.clear_chat_btn = ctk.CTkButton(
            input_frame,
            text="Clear",
            command=self.clear_chat_history,
            width=80
        )
        self.clear_chat_btn.grid(row=1, column=0, padx=(0, 5), pady=5, sticky="w")
        
        self.copy_chat_btn = ctk.CTkButton(
            input_frame,
            text="Copy",
            command=self.copy_chat_to_clipboard,
            width=80
        )
        self.copy_chat_btn.grid(row=1, column=1, padx=(5, 0), pady=5, sticky="e")
    
    def create_settings_panel(self) -> None:
        self.settings_panel = ctk.CTkFrame(self.main_container)
        self.settings_panel.grid(row=0, column=2, padx=(5, 0), pady=5, sticky="nsew")
        
        panel_header = ctk.CTkFrame(self.settings_panel, height=40)
        panel_header.pack(fill=tk.X, padx=5, pady=(5, 0))
        
        title_label = ctk.CTkLabel(
            panel_header,
            text="Settings & Controls",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        title_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        settings_container = ctk.CTkScrollableFrame(self.settings_panel)
        settings_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.create_appearance_section(settings_container)
        self.create_llm_settings_section(settings_container)
        self.create_rag_settings_section(settings_container)
        self.create_system_info_section(settings_container)
    
    def create_appearance_section(self, parent: ctk.CTkFrame) -> None:
        section_label = ctk.CTkLabel(
            parent,
            text="Appearance",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        section_label.pack(pady=(0, 10), padx=10, anchor="w")
        
        theme_frame = ctk.CTkFrame(parent)
        theme_frame.pack(fill=tk.X, padx=10, pady=5)
        
        theme_label = ctk.CTkLabel(theme_frame, text="Theme:")
        theme_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.theme_var = tk.StringVar(value=self.app_config.ui.theme)
        theme_options = ["dark", "light", "system"]
        theme_menu = ctk.CTkOptionMenu(
            theme_frame,
            values=theme_options,
            variable=self.theme_var,
            command=self.change_theme,
            width=120
        )
        theme_menu.pack(side=tk.RIGHT)
        
        font_frame = ctk.CTkFrame(parent)
        font_frame.pack(fill=tk.X, padx=10, pady=5)
        
        font_label = ctk.CTkLabel(font_frame, text=f"Font Size: {self.app_config.ui.font_size}")
        font_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.font_size_slider = ctk.CTkSlider(
            font_frame,
            from_=10,
            to=18,
            number_of_steps=8,
            command=self.change_font_size,
            width=120
        )
        self.font_size_slider.set(self.app_config.ui.font_size)
        self.font_size_slider.pack(side=tk.RIGHT)
        
        separator = ctk.CTkFrame(parent, height=1)
        separator.pack(fill=tk.X, padx=10, pady=10)
    
    def create_llm_settings_section(self, parent: ctk.CTkFrame) -> None:
        section_label = ctk.CTkLabel(
            parent,
            text="AI Model",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        section_label.pack(pady=(0, 10), padx=10, anchor="w")
        
        model_frame = ctk.CTkFrame(parent)
        model_frame.pack(fill=tk.X, padx=10, pady=5)
        
        model_label = ctk.CTkLabel(model_frame, text="Model:")
        model_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.model_var = tk.StringVar(value=self.app_config.ai.llm.model)
        model_options = ["llama2:7b", "mistral:7b", "neural-chat:7b"]
        model_menu = ctk.CTkOptionMenu(
            model_frame,
            values=model_options,
            variable=self.model_var,
            command=self.change_llm_model,
            width=120
        )
        model_menu.pack(side=tk.RIGHT)
        
        temp_frame = ctk.CTkFrame(parent)
        temp_frame.pack(fill=tk.X, padx=10, pady=5)
        
        temp_label = ctk.CTkLabel(temp_frame, text=f"Temperature: {self.app_config.ai.llm.temperature}")
        temp_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.temp_slider = ctk.CTkSlider(
            temp_frame,
            from_=0.0,
            to=1.0,
            number_of_steps=20,
            command=self.change_temperature,
            width=120
        )
        self.temp_slider.set(self.app_config.ai.llm.temperature)
        self.temp_slider.pack(side=tk.RIGHT)
        
        separator = ctk.CTkFrame(parent, height=1)
        separator.pack(fill=tk.X, padx=10, pady=10)
    
    def create_rag_settings_section(self, parent: ctk.CTkFrame) -> None:
        section_label = ctk.CTkLabel(
            parent,
            text="Search Settings",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        section_label.pack(pady=(0, 10), padx=10, anchor="w")
        
        chunk_frame = ctk.CTkFrame(parent)
        chunk_frame.pack(fill=tk.X, padx=10, pady=5)
        
        chunk_label = ctk.CTkLabel(chunk_frame, text="Chunks to Retrieve:")
        chunk_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.chunk_var = tk.StringVar(value=str(self.app_config.ai.rag.top_k))
        chunk_spinbox = tk.Spinbox(
            chunk_frame,
            from_=1,
            to=10,
            textvariable=self.chunk_var,
            width=10,
            command=self.change_chunk_count
        )
        chunk_spinbox.pack(side=tk.RIGHT)
        
        similarity_frame = ctk.CTkFrame(parent)
        similarity_frame.pack(fill=tk.X, padx=10, pady=5)
        
        similarity_label = ctk.CTkLabel(
            similarity_frame, 
            text=f"Similarity Threshold: {self.app_config.ai.rag.similarity_threshold}"
        )
        similarity_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.similarity_slider = ctk.CTkSlider(
            similarity_frame,
            from_=0.5,
            to=0.95,
            number_of_steps=10,
            command=self.change_similarity_threshold,
            width=120
        )
        self.similarity_slider.set(self.app_config.ai.rag.similarity_threshold)
        self.similarity_slider.pack(side=tk.RIGHT)
        
        separator = ctk.CTkFrame(parent, height=1)
        separator.pack(fill=tk.X, padx=10, pady=10)
    
    def create_system_info_section(self, parent: ctk.CTkFrame) -> None:
        section_label = ctk.CTkLabel(
            parent,
            text="System Information",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        section_label.pack(pady=(0, 10), padx=10, anchor="w")
        
        info_frame = ctk.CTkFrame(parent)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        version_label = ctk.CTkLabel(info_frame, text=f"Version: {self.app_config.app.version}")
        version_label.pack(anchor="w", pady=2)
        
        doc_count = self.get_document_count()
        docs_label = ctk.CTkLabel(info_frame, text=f"Documents: {doc_count}")
        docs_label.pack(anchor="w", pady=2)
        
        memory_label = ctk.CTkLabel(info_frame, text="Memory: Calculating...")
        memory_label.pack(anchor="w", pady=2)
        
        self.system_info_labels = {
            'memory': memory_label
        }
        
        self.update_system_info()
    
    def create_status_bar(self) -> None:
        self.status_bar = ctk.CTkFrame(self, height=30)
        self.status_bar.grid(row=1, column=0, columnspan=3, sticky="ew")
        
        self.status_bar.grid_columnconfigure(0, weight=1)
        
        self.status_label = ctk.CTkLabel(
            self.status_bar,
            text="Ready - DocuBot Initialized",
            font=ctk.CTkFont(size=11)
        )
        self.status_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.progress_indicator = ctk.CTkProgressBar(self.status_bar, width=100, height=3)
        self.progress_indicator.grid(row=0, column=1, padx=10, pady=5, sticky="e")
        self.progress_indicator.stop()
        
        self.progress_label = ctk.CTkLabel(
            self.status_bar,
            text="",
            font=ctk.CTkFont(size=10)
        )
        self.progress_label.grid(row=0, column=2, padx=(0, 10), pady=5, sticky="e")
    
    def setup_event_bindings(self) -> None:
        self.bind("<Control-n>", lambda e: self.new_conversation())
        self.bind("<Control-o>", lambda e: self.open_document_dialog())
        self.bind("<Control-Shift-O>", lambda e: self.open_folder_dialog())
        self.bind("<Control-s>", lambda e: self.save_conversation())
        self.bind("<Control-e>", lambda e: self.export_conversation())
        self.bind("<Control-,>", lambda e: self.open_settings_dialog())
        self.bind("<Control-t>", lambda e: self.toggle_theme())
        self.bind("<Control-plus>", lambda e: self.increase_font_size())
        self.bind("<Control-minus>", lambda e: self.decrease_font_size())
        self.bind("<F1>", lambda e: self.open_documentation())
        self.bind("<Control-q>", lambda e: self.on_closing())
        self.bind("<Configure>", self.on_window_resize)
    
    def apply_initial_settings(self) -> None:
        self.load_document_list()
        self.update_system_info()
        self.after(1000, self.start_background_tasks)
    
    def start_background_tasks(self) -> None:
        self.monitor_system_resources()
        self.after(30000, self.start_background_tasks)
    
    def update_status_bar(self, message: str, progress: float = 0.0) -> None:
        self.status_label.configure(text=message)
        
        if progress > 0.0:
            self.progress_indicator.start()
            self.progress_indicator.set(progress)
            self.progress_label.configure(text=f"{progress*100:.0f}%")
        else:
            self.progress_indicator.stop()
            self.progress_label.configure(text="")
    
    def update_system_info(self) -> None:
        if hasattr(self, 'system_info_labels'):
            try:
                import psutil
                memory = psutil.virtual_memory()
                memory_usage = f"{memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB"
                self.system_info_labels['memory'].configure(text=f"Memory: {memory_usage}")
            except ImportError:
                self.system_info_labels['memory'].configure(text="Memory: psutil not available")
    
    def monitor_system_resources(self) -> None:
        self.update_system_info()
    
    def load_document_list(self) -> None:
        if not self.db_client:
            return
        
        try:
            documents = self.db_client.get_all_documents()
            self.documents_listbox.delete(0, tk.END)
            
            for doc in documents:
                display_name = doc.get('file_name', 'Unknown')
                if len(display_name) > 30:
                    display_name = display_name[:27] + "..."
                self.documents_listbox.insert(tk.END, display_name)
            
            self.document_count_label.configure(text=f"({len(documents)})")
            self.active_documents = documents
            
        except Exception as e:
            self.logger.error(f"Failed to load document list: {e}")
            self.show_error_dialog("Document Load Error", str(e))
    
    def get_document_count(self) -> int:
        if not self.db_client:
            return 0
        
        try:
            return self.db_client.get_document_count()
        except:
            return 0
    
    def upload_document(self) -> None:
        from tkinter import filedialog
        
        file_paths = filedialog.askopenfilenames(
            title="Select Documents to Upload",
            filetypes=[
                ("All supported formats", "*.pdf *.docx *.txt *.md *.epub *.html *.csv"),
                ("PDF files", "*.pdf"),
                ("Word documents", "*.docx"),
                ("Text files", "*.txt *.md"),
                ("E-books", "*.epub"),
                ("Web pages", "*.html"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if file_paths:
            self.process_documents(file_paths)
    
    def process_documents(self, file_paths: List[str]) -> None:
        if not file_paths:
            return
        
        self.update_status_bar(f"Processing {len(file_paths)} document(s)...", 0.1)
        self.is_processing = True
        
        def process_in_background():
            try:
                for i, file_path in enumerate(file_paths):
                    progress = (i + 1) / len(file_paths)
                    self.after(0, self.update_status_bar, 
                             f"Processing: {os.path.basename(file_path)}...", 
                             progress * 0.8)
                    
                    if self.document_processor:
                        result = self.document_processor.process_document(file_path)
                        
                        if result.get('status') == 'success':
                            self.after(0, self.append_to_chat, 
                                     f"Processed: {os.path.basename(file_path)}\n", 
                                     "system")
                        else:
                            self.after(0, self.append_to_chat, 
                                     f"Failed: {os.path.basename(file_path)} - {result.get('error', 'Unknown error')}\n", 
                                     "error")
                
                self.after(0, self.update_status_bar, "Documents processed successfully", 1.0)
                self.after(0, self.load_document_list)
                self.after(1000, lambda: self.update_status_bar("Ready", 0.0))
                
            except Exception as e:
                self.logger.error(f"Document processing error: {e}")
                self.after(0, self.show_error_dialog, "Processing Error", str(e))
                self.after(0, self.update_status_bar, "Document processing failed", 0.0)
            
            finally:
                self.is_processing = False
        
        process_thread = threading.Thread(target=process_in_background, daemon=True)
        process_thread.start()
    
    def remove_selected_document(self) -> None:
        selection = self.documents_listbox.curselection()
        if not selection:
            self.show_info_dialog("No Selection", "Please select a document to remove.")
            return
        
        index = selection[0]
        if index < len(self.active_documents):
            document = self.active_documents[index]
            
            confirm = self.ask_yes_no(
                "Confirm Removal",
                f"Remove document '{document.get('file_name')}'?\n\n"
                "This will delete the document from the database but keep the original file."
            )
            
            if confirm:
                try:
                    if self.db_client:
                        self.db_client.delete_document(document['id'])
                        self.load_document_list()
                        self.update_status_bar(f"Removed: {document.get('file_name')}")
                except Exception as e:
                    self.logger.error(f"Failed to remove document: {e}")
                    self.show_error_dialog("Removal Error", str(e))
    
    def refresh_document_list(self) -> None:
        self.update_status_bar("Refreshing document list...", 0.5)
        self.load_document_list()
        self.update_status_bar("Document list refreshed", 1.0)
        self.after(1000, lambda: self.update_status_bar("Ready", 0.0))
    
    def on_document_selected(self, event) -> None:
        selection = self.documents_listbox.curselection()
        if selection:
            index = selection[0]
            if index < len(self.active_documents):
                document = self.active_documents[index]
                self.update_status_bar(f"Selected: {document.get('file_name')}")
    
    def on_document_double_click(self, event) -> None:
        selection = self.documents_listbox.curselection()
        if selection:
            index = selection[0]
            if index < len(self.active_documents):
                document = self.active_documents[index]
                self.show_document_preview(document)
    
    def show_document_preview(self, document: Dict[str, Any]) -> None:
        from tkinter import Toplevel
        
        preview_window = Toplevel(self)
        preview_window.title(f"Preview: {document.get('file_name')}")
        preview_window.geometry("600x400")
        preview_window.transient(self)
        preview_window.grab_set()
        
        text_widget = tk.Text(
            preview_window,
            wrap=tk.WORD,
            bg="white" if self.app_config.ui.theme == "light" else "#2b2b2b",
            fg="black" if self.app_config.ui.theme == "light" else "white",
            font=("Segoe UI", 11)
        )
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = tk.Scrollbar(text_widget)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=text_widget.yview)
        
        try:
            if self.db_client:
                chunks = self.db_client.get_document_chunks(document['id'])
                preview_text = "\n\n---\n\n".join([chunk['text_content'] for chunk in chunks[:5]])
                text_widget.insert(1.0, preview_text[:5000])
        except Exception as e:
            text_widget.insert(1.0, f"Error loading preview: {str(e)}")
        
        text_widget.config(state=tk.DISABLED)
        
        close_btn = ctk.CTkButton(
            preview_window,
            text="Close",
            command=preview_window.destroy,
            width=100
        )
        close_btn.pack(pady=10)
    
    def process_user_query(self) -> None:
        query = self.query_entry.get().strip()
        if not query:
            return
        
        if not self.active_documents:
            self.show_info_dialog("No Documents", "Please upload documents before asking questions.")
            return
        
        if not self.core_app:
            self.show_error_dialog("System Error", "Application not properly initialized.")
            return
        
        self.query_entry.delete(0, tk.END)
        self.append_to_chat(f"You: {query}\n\n", "user")
        
        self.update_status_bar("Processing query...", 0.3)
        self.is_processing = True
        
        def process_query_in_background():
            try:
                response = self.core_app.process_query(query)
                
                self.after(0, self.append_to_chat, f"Assistant: {response.get('answer', 'No response')}\n\n", "assistant")
                
                if response.get('sources'):
                    self.after(0, self.append_to_chat, "Sources:\n", "system")
                    for source in response['sources'][:3]:
                        self.after(0, self.append_to_chat, f"• {source}\n", "source")
                    self.after(0, self.append_to_chat, "\n", "system")
                
                self.after(0, self.update_status_bar, "Query processed successfully", 1.0)
                self.after(1000, lambda: self.update_status_bar("Ready", 0.0))
                
            except Exception as e:
                self.logger.error(f"Query processing error: {e}")
                self.after(0, self.append_to_chat, f"Error: {str(e)}\n\n", "error")
                self.after(0, self.update_status_bar, "Query processing failed", 0.0)
            
            finally:
                self.is_processing = False
        
        query_thread = threading.Thread(target=process_query_in_background, daemon=True)
        query_thread.start()
    
    def append_to_chat(self, text: str, message_type: str = "normal") -> None:
        self.chat_text.configure(state=tk.NORMAL)
        
        colors = {
            "user": "#3b8ed0",
            "assistant": "#2ecc71",
            "system": "#95a5a6",
            "error": "#e74c3c",
            "source": "#f39c12",
            "normal": "default"
        }
        
        tag_name = f"tag_{message_type}"
        self.chat_text.tag_configure(tag_name, foreground=colors.get(message_type, "default"))
        
        self.chat_text.insert(tk.END, text, tag_name)
        self.chat_text.see(tk.END)
        self.chat_text.configure(state=tk.DISABLED)
    
    def clear_chat_history(self) -> None:
        self.chat_text.configure(state=tk.NORMAL)
        self.chat_text.delete(1.0, tk.END)
        self.chat_text.configure(state=tk.DISABLED)
        self.current_conversation_id = None
        self.update_status_bar("Chat cleared")
    
    def copy_chat_to_clipboard(self) -> None:
        try:
            content = self.chat_text.get(1.0, tk.END).strip()
            if content:
                self.clipboard_clear()
                self.clipboard_append(content)
                self.update_status_bar("Chat copied to clipboard")
            else:
                self.show_info_dialog("Empty Chat", "No content to copy.")
        except Exception as e:
            self.logger.error(f"Failed to copy chat: {e}")
            self.show_error_dialog("Copy Error", str(e))
    
    def new_conversation(self) -> None:
        self.clear_chat_history()
        if self.db_client:
            self.current_conversation_id = self.db_client.create_conversation("New Conversation")
        self.update_status_bar("New conversation started")
    
    def open_document_dialog(self) -> None:
        self.upload_document()
    
    def open_folder_dialog(self) -> None:
        from tkinter import filedialog
        
        folder_path = filedialog.askdirectory(title="Select Folder with Documents")
        if folder_path:
            supported_extensions = {'.pdf', '.docx', '.txt', '.md', '.epub', '.html', '.csv'}
            file_paths = []
            
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if Path(file).suffix.lower() in supported_extensions:
                        file_paths.append(os.path.join(root, file))
            
            if file_paths:
                self.process_documents(file_paths[:50])
            else:
                self.show_info_dialog("No Documents", "No supported documents found in selected folder.")
    
    def save_conversation(self) -> None:
        content = self.chat_text.get(1.0, tk.END).strip()
        if not content:
            self.show_info_dialog("Empty Conversation", "No conversation to save.")
            return
        
        from tkinter import filedialog
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                conversation_data = {
                    "timestamp": datetime.now().isoformat(),
                    "content": content,
                    "documents": [doc.get('file_name') for doc in self.active_documents]
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(conversation_data, f, indent=2, ensure_ascii=False)
                
                self.update_status_bar(f"Conversation saved to {os.path.basename(file_path)}")
                
            except Exception as e:
                self.logger.error(f"Failed to save conversation: {e}")
                self.show_error_dialog("Save Error", str(e))
    
    def export_conversation(self) -> None:
        content = self.chat_text.get(1.0, tk.END).strip()
        if not content:
            self.show_info_dialog("Empty Conversation", "No conversation to export.")
            return
        
        from tkinter import filedialog
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".md",
            filetypes=[
                ("Markdown files", "*.md"),
                ("Text files", "*.txt"),
                ("HTML files", "*.html"),
                ("PDF files", "*.pdf"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                if file_path.endswith('.md'):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(f"# DocuBot Conversation\n\n")
                        f.write(f"*Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
                        f.write(content)
                
                elif file_path.endswith('.txt'):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(f"DocuBot Conversation\n")
                        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(content)
                
                self.update_status_bar(f"Exported to {os.path.basename(file_path)}")
                
            except Exception as e:
                self.logger.error(f"Failed to export conversation: {e}")
                self.show_error_dialog("Export Error", str(e))
    
    def open_settings_dialog(self) -> None:
        try:
            from .settings_panel import SettingsDialog
            settings_dialog = SettingsDialog(self, self.app_config)
            self.wait_window(settings_dialog)
            
            if settings_dialog.result:
                self.apply_updated_settings(settings_dialog.result)
        except ImportError:
            self.show_info_dialog("Feature Unavailable", "Settings dialog module not available.")
    
    def open_preferences_dialog(self) -> None:
        self.open_settings_dialog()
    
    def open_document_manager(self) -> None:
        try:
            from .document_panel import DocumentManagerDialog
            doc_manager = DocumentManagerDialog(self, self.db_client)
            self.wait_window(doc_manager)
            
            if doc_manager.refresh_needed:
                self.load_document_list()
        except ImportError:
            self.show_info_dialog("Feature Unavailable", "Document manager module not available.")
    
    def open_vector_db_manager(self) -> None:
        self.show_info_dialog("Coming Soon", "Vector database manager will be available in a future update.")
    
    def open_model_manager(self) -> None:
        try:
            from .settings_panel import ModelManagerDialog
            model_manager = ModelManagerDialog(self, self.app_config)
            self.wait_window(model_manager)
        except ImportError:
            self.show_info_dialog("Feature Unavailable", "Model manager module not available.")
    
    def change_theme(self, choice: str) -> None:
        self.app_config.ui.theme = choice
        ctk.set_appearance_mode(choice)
        
        self.documents_listbox.configure(
            bg=self.get_listbox_background_color(),
            fg=self.get_listbox_foreground_color()
        )
        
        self.update_status_bar(f"Theme changed to {choice}")
    
    def toggle_theme(self) -> None:
        new_theme = "light" if self.app_config.ui.theme == "dark" else "dark"
        self.change_theme(new_theme)
    
    def change_font_size(self, value: float) -> None:
        font_size = int(value)
        self.app_config.ui.font_size = font_size
        
        self.chat_text.configure(font=ctk.CTkFont(size=font_size))
        self.query_entry.configure(font=ctk.CTkFont(size=font_size))
    
    def increase_font_size(self) -> None:
        if self.app_config.ui.font_size < 20:
            self.app_config.ui.font_size += 1
            self.font_size_slider.set(self.app_config.ui.font_size)
            self.change_font_size(self.app_config.ui.font_size)
    
    def decrease_font_size(self) -> None:
        if self.app_config.ui.font_size > 10:
            self.app_config.ui.font_size -= 1
            self.font_size_slider.set(self.app_config.ui.font_size)
            self.change_font_size(self.app_config.ui.font_size)
    
    def change_llm_model(self, choice: str) -> None:
        self.app_config.ai.llm.model = choice
        if self.llm_client:
            self.llm_client.set_model(choice)
        self.update_status_bar(f"Model changed to {choice}")
    
    def change_temperature(self, value: float) -> None:
        self.app_config.ai.llm.temperature = value
        if self.llm_client:
            self.llm_client.set_temperature(value)
    
    def change_chunk_count(self) -> None:
        try:
            count = int(self.chunk_var.get())
            if 1 <= count <= 10:
                self.app_config.ai.rag.top_k = count
            else:
                self.chunk_var.set(str(self.app_config.ai.rag.top_k))
        except ValueError:
            self.chunk_var.set(str(self.app_config.ai.rag.top_k))
    
    def change_similarity_threshold(self, value: float) -> None:
        self.app_config.ai.rag.similarity_threshold = value
    
    def apply_updated_settings(self, updated_config: AppConfig) -> None:
        self.app_config = updated_config
        self.initialize_application_components()
        self.load_document_list()
        self.update_status_bar("Settings applied successfully")
    
    def reset_ui_layout(self) -> None:
        self.documents_listbox.pack_forget()
        self.chat_text.grid_forget()
        
        self.create_document_panel()
        self.create_chat_panel()
        
        self.update_status_bar("UI layout reset")
    
    def show_keyboard_shortcuts(self) -> None:
        shortcuts = """
Keyboard Shortcuts:

File Operations:
  Ctrl+N      - New Conversation
  Ctrl+O      - Open Document
  Ctrl+Shift+O - Open Folder
  Ctrl+S      - Save Conversation
  Ctrl+E      - Export Conversation
  Ctrl+Q      - Exit

View Operations:
  Ctrl+T      - Toggle Theme
  Ctrl++      - Increase Font Size
  Ctrl+-      - Decrease Font Size

Application:
  F1          - Documentation
  Ctrl+,      - Settings
  Enter       - Send Message (in chat)
        """
        
        self.show_info_dialog("Keyboard Shortcuts", shortcuts.strip())
    
    def open_documentation(self) -> None:
        import webbrowser
        
        try:
            webbrowser.open("https://github.com/yourusername/docubot/docs")
            self.update_status_bar("Opening documentation...")
        except Exception as e:
            self.logger.error(f"Failed to open documentation: {e}")
            self.show_error_dialog("Browser Error", "Could not open documentation in browser.")
    
    def check_for_updates(self) -> None:
        self.update_status_bar("Checking for updates...", 0.5)
        
        def check_in_background():
            try:
                import requests
                response = requests.get(
                    "https://api.github.com/repos/yourusername/docubot/releases/latest", 
                    timeout=5
                )
                if response.status_code == 200:
                    latest_version = response.json()['tag_name']
                    current_version = self.app_config.app.version
                    
                    if latest_version != current_version:
                        self.after(0, self.show_info_dialog, "Update Available", 
                                 f"Version {latest_version} is available.\nCurrent version: {current_version}")
                    else:
                        self.after(0, self.show_info_dialog, "Up to Date", 
                                 "You have the latest version.")
                else:
                    self.after(0, self.show_error_dialog, "Update Check Failed", 
                             "Could not check for updates.")
            
            except Exception as e:
                self.logger.error(f"Update check failed: {e}")
                self.after(0, self.show_error_dialog, "Update Error", str(e))
            
            finally:
                self.after(0, self.update_status_bar, "Ready", 0.0)
        
        update_thread = threading.Thread(target=check_in_background, daemon=True)
        update_thread.start()
    
    def show_about_dialog(self) -> None:
        about_text = f"""
DocuBot - Personal AI Knowledge Assistant
Version: {self.app_config.app.version}

A 100% local AI assistant for your documents.
Transforms your document collection into a digital second brain.

Features:
• Privacy First - All data remains on your device
• Zero Subscription - Free forever, open source
• Offline Native - Works without internet connection
• Resource Efficient - Lightweight for home laptops/PCs

© 2026 DocuBot Team. All rights reserved.
        """
        
        self.show_info_dialog("About DocuBot", about_text.strip())
    
    def show_info_dialog(self, title: str, message: str) -> None:
        from tkinter import messagebox
        messagebox.showinfo(title, message)
    
    def show_error_dialog(self, title: str, message: str) -> None:
        from tkinter import messagebox
        messagebox.showerror(title, message)
    
    def ask_yes_no(self, title: str, message: str) -> bool:
        from tkinter import messagebox
        return messagebox.askyesno(title, message)
    
    def on_window_resize(self, event) -> None:
        pass
    
    def on_closing(self) -> None:
        if self.is_processing:
            response = self.ask_yes_no(
                "Processing in Progress",
                "Document processing is still in progress.\nDo you want to quit anyway?"
            )
            if not response:
                return
        
        self.update_status_bar("Shutting down...")
        
        try:
            if self.app_config.storage.backup_enabled:
                self.create_backup()
        except Exception as e:
            self.logger.error(f"Backup failed during shutdown: {e}")
        
        self.destroy()
    
    def create_backup(self) -> bool:
        try:
            backup_dir = os.path.join(os.path.expanduser("~"), ".docubot", "backups")
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(backup_dir, f"backup_{timestamp}.json")
            
            backup_data = {
                'timestamp': timestamp,
                'documents_count': len(self.active_documents),
                'app_version': self.app_config.app.version
            }
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2)
            
            self.logger.info(f"Backup created: {backup_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            return False


def run_application(config: Optional[AppConfig] = None) -> None:
    try:
        print("=" * 60)
        print("DOCUBOT - Starting Application")
        print("=" * 60)
        print(f"Python version: {sys.version}")
        print(f"Current directory: {os.getcwd()}")
        print("=" * 60)
        
        app = MainWindow(config)
        app.mainloop()
        
        print("\nDocuBot application closed successfully.")
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            import tkinter.messagebox as messagebox
            messagebox.showerror(
                "Fatal Error", 
                f"DocuBot encountered a fatal error:\n\n{str(e)}\n\n"
                "Please check the console for details."
            )
        except:
            pass
        
        sys.exit(1)


if __name__ == "__main__":
    run_application()