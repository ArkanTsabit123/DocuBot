"""
Dark Theme Configuration
"""

DARK_THEME = {
    'name': 'dark',
    'bg_color': '#1E1E1E',
    'fg_color': '#2B2B2B',
    'text_color': '#FFFFFF',
    'accent_color': '#2B5278',
    'secondary_accent': '#1E6B5E',
    'border_color': '#404040',
    'hover_color': '#333333',
    'scrollbar_color': '#555555',
    'success_color': '#1E6B5E',
    'warning_color': '#8B4513',
    'error_color': '#8B0000',
    'info_color': '#2B5278',
    
    'fonts': {
        'default': ('Segoe UI', 11),
        'heading': ('Segoe UI', 14, 'bold'),
        'title': ('Segoe UI', 16, 'bold'),
        'monospace': ('Consolas', 10)
    }
}


class DarkTheme:
    """Dark theme implementation"""
    
    @staticmethod
    def apply(window):
        """Apply dark theme to window"""
        import customtkinter as ctk
        
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Additional theme-specific configurations
        window.configure(fg_color=DARK_THEME['bg_color'])
    
    @staticmethod
    def get_colors():
        """Get theme colors"""
        return DARK_THEME
    
    @staticmethod
    def get_font(font_type='default'):
        """Get theme font"""
        return DARK_THEME['fonts'].get(font_type, DARK_THEME['fonts']['default'])
