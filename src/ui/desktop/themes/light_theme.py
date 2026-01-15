"""
Light Theme Configuration
"""

LIGHT_THEME = {
    'name': 'light',
    'bg_color': '#F0F0F0',
    'fg_color': '#FFFFFF',
    'text_color': '#000000',
    'accent_color': '#2196F3',
    'secondary_accent': '#4CAF50',
    'border_color': '#E0E0E0',
    'hover_color': '#F5F5F5',
    'scrollbar_color': '#CCCCCC',
    'success_color': '#4CAF50',
    'warning_color': '#FF9800',
    'error_color': '#F44336',
    'info_color': '#2196F3',
    
    'fonts': {
        'default': ('Segoe UI', 11),
        'heading': ('Segoe UI', 14, 'bold'),
        'title': ('Segoe UI', 16, 'bold'),
        'monospace': ('Consolas', 10)
    }
}


class LightTheme:
    """Light theme implementation"""
    
    @staticmethod
    def apply(window):
        """Apply light theme to window"""
        import customtkinter as ctk
        
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        
        # Additional theme-specific configurations
        window.configure(fg_color=LIGHT_THEME['bg_color'])
    
    @staticmethod
    def get_colors():
        """Get theme colors"""
        return LIGHT_THEME
    
    @staticmethod
    def get_font(font_type='default'):
        """Get theme font"""
        return LIGHT_THEME['fonts'].get(font_type, LIGHT_THEME['fonts']['default'])
