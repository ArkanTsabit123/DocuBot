"""
System Theme Configuration
Follows system theme settings
"""

import platform


class SystemTheme:
    """System theme implementation"""
    
    @staticmethod
    def apply(window):
        """Apply system theme to window"""
        import customtkinter as ctk
        
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")
    
    @staticmethod
    def get_colors():
        """Get current system colors"""
        # This would detect system theme and return appropriate colors
        import customtkinter as ctk
        
        current_mode = ctk.get_appearance_mode()
        
        if current_mode == "dark":
            from .dark_theme import DARK_THEME
            return DARK_THEME
        else:
            from .light_theme import LIGHT_THEME
            return LIGHT_THEME
    
    @staticmethod
    def get_font(font_type='default'):
        """Get theme font"""
        colors = SystemTheme.get_colors()
        return colors['fonts'].get(font_type, colors['fonts']['default'])
    
    @staticmethod
    def is_dark_mode() -> bool:
        """Check if system is in dark mode"""
        try:
            import customtkinter as ctk
            return ctk.get_appearance_mode() == "dark"
        except:
            return False
    
    @staticmethod
    def get_system_info() -> dict:
        """Get system information"""
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'theme_support': 'native' if platform.system() in ['Windows', 'Darwin'] else 'basic'
        }
