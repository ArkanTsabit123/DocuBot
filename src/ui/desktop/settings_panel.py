"""
Settings Panel
Application configuration and preferences
"""

import customtkinter as ctk
from typing import Dict, Any, Optional, Callable
import json
from pathlib import Path
from ..core.config import ConfigManager, get_config
from ..core.logger import get_logger

logger = get_logger(__name__)


class SettingsPanel(ctk.CTkFrame):
    """Settings panel for application configuration"""
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load()
        self.unsaved_changes = False
        
        self._create_widgets()
        self._setup_layout()
        self._load_current_settings()
        
        logger.debug("Settings panel initialized")
    
    def _create_widgets(self):
        """Create settings panel widgets"""
        
        # Notebook-style tabs
        self.tabview = ctk.CTkTabview(self)
        self.general_tab = self.tabview.add("General")
        self.ai_tab = self.tabview.add("AI")
        self.document_tab = self.tabview.add("Documents")
        self.ui_tab = self.tabview.add("UI")
        self.advanced_tab = self.tabview.add("Advanced")
        
        # General settings
        self._create_general_tab()
        
        # AI settings
        self._create_ai_tab()
        
        # Document settings
        self._create_document_tab()
        
        # UI settings
        self._create_ui_tab()
        
        # Advanced settings
        self._create_advanced_tab()
        
        # Action buttons
        self.button_frame = ctk.CTkFrame(self)
        
        self.save_button = ctk.CTkButton(
            self.button_frame,
            text="Save Settings",
            width=120,
            command=self._save_settings
        )
        
        self.reset_button = ctk.CTkButton(
            self.button_frame,
            text="Reset to Defaults",
            width=120,
            fg_color="gray",
            hover_color="#555555",
            command=self._reset_settings
        )
        
        self.cancel_button = ctk.CTkButton(
            self.button_frame,
            text="Cancel",
            width=80,
            fg_color="#8B0000",
            hover_color="#6A0000",
            command=self._cancel_changes
        )
    
    def _create_general_tab(self):
        """Create general settings tab"""
        
        # Application settings
        app_label = ctk.CTkLabel(
            self.general_tab,
            text="Application Settings",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        app_label.pack(anchor="w", padx=20, pady=(20, 10))
        
        # Auto-save
        self.auto_save_var = ctk.BooleanVar(value=True)
        self.auto_save_check = ctk.CTkCheckBox(
            self.general_tab,
            text="Enable auto-save",
            variable=self.auto_save_var,
            command=self._mark_unsaved
        )
        self.auto_save_check.pack(anchor="w", padx=40, pady=5)
        
        # Auto-save interval
        self.auto_save_frame = ctk.CTkFrame(self.general_tab, fg_color="transparent")
        
        auto_save_label = ctk.CTkLabel(
            self.auto_save_frame,
            text="Auto-save interval (minutes):"
        )
        auto_save_label.pack(side="left", padx=(0, 10))
        
        self.auto_save_slider = ctk.CTkSlider(
            self.auto_save_frame,
            from_=1,
            to=60,
            number_of_steps=59,
            command=self._mark_unsaved
        )
        self.auto_save_slider.pack(side="left", fill="x", expand=True)
        
        self.auto_save_value = ctk.CTkLabel(
            self.auto_save_frame,
            text="30",
            width=30
        )
        self.auto_save_value.pack(side="left", padx=(10, 0))
        
        self.auto_save_frame.pack(anchor="w", padx=40, pady=5, fill="x")
        
        # Privacy settings
        privacy_label = ctk.CTkLabel(
            self.general_tab,
            text="Privacy",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        privacy_label.pack(anchor="w", padx=20, pady=(20, 10))
        
        self.telemetry_var = ctk.BooleanVar(value=False)
        self.telemetry_check = ctk.CTkCheckBox(
            self.general_tab,
            text="Send anonymous usage statistics",
            variable=self.telemetry_var,
            command=self._mark_unsaved
        )
        self.telemetry_check.pack(anchor="w", padx=40, pady=5)
        
        self.crash_reports_var = ctk.BooleanVar(value=False)
        self.crash_reports_check = ctk.CTkCheckBox(
            self.general_tab,
            text="Send crash reports",
            variable=self.crash_reports_var,
            command=self._mark_unsaved
        )
        self.crash_reports_check.pack(anchor="w", padx=40, pady=5)
    
    def _create_ai_tab(self):
        """Create AI settings tab"""
        
        # LLM settings
        llm_label = ctk.CTkLabel(
            self.ai_tab,
            text="Language Model",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        llm_label.pack(anchor="w", padx=20, pady=(20, 10))
        
        # Model selection
        model_frame = ctk.CTkFrame(self.ai_tab, fg_color="transparent")
        
        model_label = ctk.CTkLabel(
            model_frame,
            text="Model:"
        )
        model_label.pack(side="left", padx=(0, 10))
        
        self.model_combo = ctk.CTkComboBox(
            model_frame,
            values=["llama2:7b", "mistral:7b", "neural-chat:7b"],
            command=self._mark_unsaved
        )
        self.model_combo.pack(side="left", fill="x", expand=True)
        
        model_frame.pack(anchor="w", padx=40, pady=5, fill="x")
        
        # Temperature
        temp_frame = ctk.CTkFrame(self.ai_tab, fg_color="transparent")
        
        temp_label = ctk.CTkLabel(
            temp_frame,
            text="Temperature:"
        )
        temp_label.pack(side="left", padx=(0, 10))
        
        self.temp_slider = ctk.CTkSlider(
            temp_frame,
            from_=0,
            to=1,
            number_of_steps=100,
            command=self._update_temp_label
        )
        self.temp_slider.pack(side="left", fill="x", expand=True)
        
        self.temp_value = ctk.CTkLabel(
            temp_frame,
            text="0.1",
            width=40
        )
        self.temp_value.pack(side="left", padx=(10, 0))
        
        temp_frame.pack(anchor="w", padx=40, pady=5, fill="x")
        
        # Max tokens
        tokens_frame = ctk.CTkFrame(self.ai_tab, fg_color="transparent")
        
        tokens_label = ctk.CTkLabel(
            tokens_frame,
            text="Max tokens:"
        )
        tokens_label.pack(side="left", padx=(0, 10))
        
        self.tokens_entry = ctk.CTkEntry(
            tokens_frame,
            width=100
        )
        self.tokens_entry.pack(side="left")
        self.tokens_entry.bind("<KeyRelease>", lambda e: self._mark_unsaved())
        
        tokens_frame.pack(anchor="w", padx=40, pady=5)
        
        # Embedding settings
        embed_label = ctk.CTkLabel(
            self.ai_tab,
            text="Embeddings",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        embed_label.pack(anchor="w", padx=20, pady=(20, 10))
        
        # Embedding model
        embed_model_frame = ctk.CTkFrame(self.ai_tab, fg_color="transparent")
        
        embed_model_label = ctk.CTkLabel(
            embed_model_frame,
            text="Model:"
        )
        embed_model_label.pack(side="left", padx=(0, 10))
        
        self.embed_model_combo = ctk.CTkComboBox(
            embed_model_frame,
            values=["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
            command=self._mark_unsaved
        )
        self.embed_model_combo.pack(side="left", fill="x", expand=True)
        
        embed_model_frame.pack(anchor="w", padx=40, pady=5, fill="x")
    
    def _create_document_tab(self):
        """Create document settings tab"""
        
        # Processing settings
        process_label = ctk.CTkLabel(
            self.document_tab,
            text="Document Processing",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        process_label.pack(anchor="w", padx=20, pady=(20, 10))
        
        # Chunk size
        chunk_frame = ctk.CTkFrame(self.document_tab, fg_color="transparent")
        
        chunk_label = ctk.CTkLabel(
            chunk_frame,
            text="Chunk size (tokens):"
        )
        chunk_label.pack(side="left", padx=(0, 10))
        
        self.chunk_size_slider = ctk.CTkSlider(
            chunk_frame,
            from_=100,
            to=1000,
            number_of_steps=18,
            command=self._update_chunk_label
        )
        self.chunk_size_slider.pack(side="left", fill="x", expand=True)
        
        self.chunk_size_value = ctk.CTkLabel(
            chunk_frame,
            text="500",
            width=40
        )
        self.chunk_size_value.pack(side="left", padx=(10, 0))
        
        chunk_frame.pack(anchor="w", padx=40, pady=5, fill="x")
        
        # Chunk overlap
        overlap_frame = ctk.CTkFrame(self.document_tab, fg_color="transparent")
        
        overlap_label = ctk.CTkLabel(
            overlap_frame,
            text="Chunk overlap:"
        )
        overlap_label.pack(side="left", padx=(0, 10))
        
        self.overlap_slider = ctk.CTkSlider(
            overlap_frame,
            from_=0,
            to=200,
            number_of_steps=40,
            command=self._update_overlap_label
        )
        self.overlap_slider.pack(side="left", fill="x", expand=True)
        
        self.overlap_value = ctk.CTkLabel(
            overlap_frame,
            text="50",
            width=40
        )
        self.overlap_value.pack(side="left", padx=(10, 0))
        
        overlap_frame.pack(anchor="w", padx=40, pady=5, fill="x")
        
        # Max file size
        size_frame = ctk.CTkFrame(self.document_tab, fg_color="transparent")
        
        size_label = ctk.CTkLabel(
            size_frame,
            text="Max file size (MB):"
        )
        size_label.pack(side="left", padx=(0, 10))
        
        self.max_size_entry = ctk.CTkEntry(
            size_frame,
            width=100
        )
        self.max_size_entry.pack(side="left")
        self.max_size_entry.bind("<KeyRelease>", lambda e: self._mark_unsaved())
        
        size_frame.pack(anchor="w", padx=40, pady=5)
        
        # OCR settings
        ocr_label = ctk.CTkLabel(
            self.document_tab,
            text="OCR Processing",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        ocr_label.pack(anchor="w", padx=20, pady=(20, 10))
        
        self.ocr_enabled_var = ctk.BooleanVar(value=False)
        self.ocr_check = ctk.CTkCheckBox(
            self.document_tab,
            text="Enable OCR for image processing",
            variable=self.ocr_enabled_var,
            command=self._mark_unsaved
        )
        self.ocr_check.pack(anchor="w", padx=40, pady=5)
    
    def _create_ui_tab(self):
        """Create UI settings tab"""
        
        # Theme settings
        theme_label = ctk.CTkLabel(
            self.ui_tab,
            text="Appearance",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        theme_label.pack(anchor="w", padx=20, pady=(20, 10))
        
        # Theme selection
        theme_frame = ctk.CTkFrame(self.ui_tab, fg_color="transparent")
        
        theme_option_label = ctk.CTkLabel(
            theme_frame,
            text="Theme:"
        )
        theme_option_label.pack(side="left", padx=(0, 10))
        
        self.theme_combo = ctk.CTkComboBox(
            theme_frame,
            values=["dark", "light", "system"],
            command=self._mark_unsaved
        )
        self.theme_combo.pack(side="left", fill="x", expand=True)
        
        theme_frame.pack(anchor="w", padx=40, pady=5, fill="x")
        
        # Language selection
        lang_frame = ctk.CTkFrame(self.ui_tab, fg_color="transparent")
        
        lang_label = ctk.CTkLabel(
            lang_frame,
            text="Language:"
        )
        lang_label.pack(side="left", padx=(0, 10))
        
        self.lang_combo = ctk.CTkComboBox(
            lang_frame,
            values=["en", "id"],
            command=self._mark_unsaved
        )
        self.lang_combo.pack(side="left", fill="x", expand=True)
        
        lang_frame.pack(anchor="w", padx=40, pady=5, fill="x")
        
        # Font size
        font_frame = ctk.CTkFrame(self.ui_tab, fg_color="transparent")
        
        font_label = ctk.CTkLabel(
            font_frame,
            text="Font size:"
        )
        font_label.pack(side="left", padx=(0, 10))
        
        self.font_slider = ctk.CTkSlider(
            font_frame,
            from_=8,
            to=20,
            number_of_steps=12,
            command=self._update_font_label
        )
        self.font_slider.pack(side="left", fill="x", expand=True)
        
        self.font_value = ctk.CTkLabel(
            font_frame,
            text="12",
            width=30
        )
        self.font_value.pack(side="left", padx=(10, 0))
        
        font_frame.pack(anchor="w", padx=40, pady=5, fill="x")
        
        # UI animations
        self.animations_var = ctk.BooleanVar(value=True)
        self.animations_check = ctk.CTkCheckBox(
            self.ui_tab,
            text="Enable UI animations",
            variable=self.animations_var,
            command=self._mark_unsaved
        )
        self.animations_check.pack(anchor="w", padx=40, pady=5)
    
    def _create_advanced_tab(self):
        """Create advanced settings tab"""
        
        # Performance settings
        perf_label = ctk.CTkLabel(
            self.advanced_tab,
            text="Performance",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        perf_label.pack(anchor="w", padx=20, pady=(20, 10))
        
        # Worker threads
        workers_frame = ctk.CTkFrame(self.advanced_tab, fg_color="transparent")
        
        workers_label = ctk.CTkLabel(
            workers_frame,
            text="Max worker threads:"
        )
        workers_label.pack(side="left", padx=(0, 10))
        
        self.workers_slider = ctk.CTkSlider(
            workers_frame,
            from_=1,
            to=8,
            number_of_steps=7,
            command=self._update_workers_label
        )
        self.workers_slider.pack(side="left", fill="x", expand=True)
        
        self.workers_value = ctk.CTkLabel(
            workers_frame,
            text="4",
            width=30
        )
        self.workers_value.pack(side="left", padx=(10, 0))
        
        workers_frame.pack(anchor="w", padx=40, pady=5, fill="x")
        
        # Cache settings
        cache_label = ctk.CTkLabel(
            self.advanced_tab,
            text="Cache",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        cache_label.pack(anchor="w", padx=20, pady=(20, 10))
        
        self.cache_enabled_var = ctk.BooleanVar(value=True)
        self.cache_check = ctk.CTkCheckBox(
            self.advanced_tab,
            text="Enable caching",
            variable=self.cache_enabled_var,
            command=self._mark_unsaved
        )
        self.cache_check.pack(anchor="w", padx=40, pady=5)
        
        # Cache size
        cache_size_frame = ctk.CTkFrame(self.advanced_tab, fg_color="transparent")
        
        cache_size_label = ctk.CTkLabel(
            cache_size_frame,
            text="Cache size (MB):"
        )
        cache_size_label.pack(side="left", padx=(0, 10))
        
        self.cache_size_entry = ctk.CTkEntry(
            cache_size_frame,
            width=100
        )
        self.cache_size_entry.pack(side="left")
        self.cache_size_entry.bind("<KeyRelease>", lambda e: self._mark_unsaved())
        
        cache_size_frame.pack(anchor="w", padx=40, pady=5)
        
        # Storage settings
        storage_label = ctk.CTkLabel(
            self.advanced_tab,
            text="Storage",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        storage_label.pack(anchor="w", padx=20, pady=(20, 10))
        
        self.backup_enabled_var = ctk.BooleanVar(value=True)
        self.backup_check = ctk.CTkCheckBox(
            self.advanced_tab,
            text="Enable automatic backups",
            variable=self.backup_enabled_var,
            command=self._mark_unsaved
        )
        self.backup_check.pack(anchor="w", padx=40, pady=5)
        
        # Auto cleanup
        cleanup_frame = ctk.CTkFrame(self.advanced_tab, fg_color="transparent")
        
        cleanup_label = ctk.CTkLabel(
            cleanup_frame,
            text="Auto-cleanup after (days):"
        )
        cleanup_label.pack(side="left", padx=(0, 10))
        
        self.cleanup_entry = ctk.CTkEntry(
            cleanup_frame,
            width=100
        )
        self.cleanup_entry.pack(side="left")
        self.cleanup_entry.bind("<KeyRelease>", lambda e: self._mark_unsaved())
        
        cleanup_frame.pack(anchor="w", padx=40, pady=5)
    
    def _setup_layout(self):
        """Arrange settings panel widgets"""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.button_frame.grid(row=1, column=0, sticky="e", padx=10, pady=(0, 10))
        self.cancel_button.pack(side="right", padx=(5, 0))
        self.reset_button.pack(side="right", padx=(5, 0))
        self.save_button.pack(side="right")
    
    def _load_current_settings(self):
        """Load current configuration into UI"""
        
        # General settings
        self.auto_save_var.set(True)  # Default to true
        self.auto_save_slider.set(30)
        self.telemetry_var.set(False)
        self.crash_reports_var.set(False)
        
        # AI settings
        self.model_combo.set(self.config.llm_model)
        self.temp_slider.set(self.config.llm_temperature)
        self.temp_value.configure(text=f"{self.config.llm_temperature:.1f}")
        self.tokens_entry.insert(0, str(self.config.llm_max_tokens))
        self.embed_model_combo.set(self.config.embedding_model)
        
        # Document settings
        self.chunk_size_slider.set(self.config.chunk_size)
        self.chunk_size_value.configure(text=str(self.config.chunk_size))
        self.overlap_slider.set(self.config.chunk_overlap)
        self.overlap_value.configure(text=str(self.config.chunk_overlap))
        self.max_size_entry.insert(0, str(self.config.max_file_size_mb))
        self.ocr_enabled_var.set(False)  # Default to false
        
        # UI settings
        self.theme_combo.set(self.config.ui_theme)
        self.lang_combo.set(self.config.ui_language)
        self.font_slider.set(self.config.ui_font_size)
        self.font_value.configure(text=str(self.config.ui_font_size))
        self.animations_var.set(True)
        
        # Advanced settings
        self.workers_slider.set(4)
        self.workers_value.configure(text="4")
        self.cache_enabled_var.set(True)
        self.cache_size_entry.insert(0, "500")
        self.backup_enabled_var.set(True)
        self.cleanup_entry.insert(0, "90")
    
    def _mark_unsaved(self, *args):
        """Mark settings as unsaved"""
        self.unsaved_changes = True
    
    def _update_temp_label(self, value):
        """Update temperature label"""
        self.temp_value.configure(text=f"{value:.1f}")
        self._mark_unsaved()
    
    def _update_chunk_label(self, value):
        """Update chunk size label"""
        self.chunk_size_value.configure(text=str(int(value)))
        self._mark_unsaved()
    
    def _update_overlap_label(self, value):
        """Update overlap label"""
        self.overlap_value.configure(text=str(int(value)))
        self._mark_unsaved()
    
    def _update_font_label(self, value):
        """Update font size label"""
        self.font_value.configure(text=str(int(value)))
        self._mark_unsaved()
    
    def _update_workers_label(self, value):
        """Update workers label"""
        self.workers_value.configure(text=str(int(value)))
        self._mark_unsaved()
    
    def _save_settings(self):
        """Save current settings"""
        try:
            # Update config from UI values
            self.config.llm_model = self.model_combo.get()
            self.config.llm_temperature = self.temp_slider.get()
            self.config.llm_max_tokens = int(self.tokens_entry.get())
            self.config.embedding_model = self.embed_model_combo.get()
            
            self.config.chunk_size = int(self.chunk_size_slider.get())
            self.config.chunk_overlap = int(self.overlap_slider.get())
            self.config.max_file_size_mb = int(self.max_size_entry.get())
            
            self.config.ui_theme = self.theme_combo.get()
            self.config.ui_language = self.lang_combo.get()
            self.config.ui_font_size = int(self.font_slider.get())
            
            # Save to file
            if self.config_manager.save():
                self.unsaved_changes = False
                
                # Apply theme change immediately
                import customtkinter as ctk
                ctk.set_appearance_mode(self.config.ui_theme)
                
                logger.info("Settings saved successfully")
                
                # Show success message
                self._show_message("Settings saved successfully", "info")
            else:
                self._show_message("Failed to save settings", "error")
                
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            self._show_message(f"Error: {str(e)}", "error")
    
    def _reset_settings(self):
        """Reset settings to defaults"""
        # This would reset to hardcoded defaults
        self._load_current_settings()
        self.unsaved_changes = True
        logger.info("Settings reset to defaults")
    
    def _cancel_changes(self):
        """Cancel unsaved changes"""
        if self.unsaved_changes:
            # Show confirmation dialog
            self._load_current_settings()
            self.unsaved_changes = False
            logger.info("Changes cancelled")
    
    def _show_message(self, message: str, message_type: str = "info"):
        """Show status message"""
        # This would show a toast or status message
        print(f"[{message_type.upper()}] {message}")
    
    def apply_theme(self):
        """Apply current theme settings"""
        import customtkinter as ctk
        ctk.set_appearance_mode(self.config.ui_theme)
    
    def get_unsaved_changes(self) -> bool:
        """Check if there are unsaved changes"""
        return self.unsaved_changes


class SettingsDialog(ctk.CTkToplevel):
    """Settings dialog window"""
    
    def __init__(self, parent):
        super().__init__(parent)
        
        self.title("DocuBot Settings")
        self.geometry("800x600")
        
        # Make dialog modal
        self.transient(parent)
        self.grab_set()
        
        self.settings_panel = SettingsPanel(self)
        self.settings_panel.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - (self.winfo_width() // 2)
        y = parent.winfo_y() + (parent.winfo_height() // 2) - (self.winfo_height() // 2)
        self.geometry(f"+{x}+{y}")
    
    def close(self):
        """Close settings dialog"""
        if self.settings_panel.get_unsaved_changes():
            # Ask for confirmation
            pass
        self.grab_release()
        self.destroy()
