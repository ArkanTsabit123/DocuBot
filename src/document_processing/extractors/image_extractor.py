"""
Image Document Extractor with OCR Support
Extracts text from images using Tesseract OCR
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError as e:
    OCR_AVAILABLE = False
    pytesseract = None
    Image = None

from .base_extractor import BaseExtractor, ExtractionResult

logger = logging.getLogger(__name__)

class ImageExtractor(BaseExtractor):
    """Extractor for image files using Tesseract OCR"""
    
    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.ocr_languages = self.config.get('ocr_languages', ['eng'])
        self.ocr_config = self.config.get('ocr_config', {})
        self.check_tesseract_availability()
    
    def check_tesseract_availability(self) -> bool:
        """Check if Tesseract is available"""
        if not OCR_AVAILABLE:
            logger.warning("OCR dependencies not installed. Install pytesseract and Pillow.")
            return False
        
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR is available")
            return True
        except Exception as e:
            logger.error(f"Tesseract not found: {e}")
            logger.info("Install Tesseract OCR: https://github.com/tesseract-ocr/tesseract")
            return False
    
    def extract(self, file_path: str) -> ExtractionResult:
        """
        Extract text from image file using OCR
        
        Args:
            file_path: Path to image file
            
        Returns:
            ExtractionResult containing extracted text and metadata
        """
        if not OCR_AVAILABLE:
            return ExtractionResult(
                content="OCR dependencies not available. Install pytesseract and Pillow.",
                metadata={'error': 'OCR not available'},
                success=False
            )
        
        try:
            if not self.check_tesseract_availability():
                return ExtractionResult(
                    content="Tesseract OCR not found on system.",
                    metadata={'error': 'Tesseract not found'},
                    success=False
                )
            
            logger.info(f"Processing image: {file_path}")
            
            image = Image.open(file_path)
            
            ocr_config = self._get_ocr_config()
            
            extracted_text = pytesseract.image_to_string(
                image,
                lang='+'.join(self.ocr_languages),
                config=ocr_config
            )
            
            metadata = self._extract_metadata(image, file_path)
            
            logger.info(f"Successfully extracted {len(extracted_text)} characters from {file_path}")
            
            return ExtractionResult(
                content=extracted_text,
                metadata=metadata,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error extracting text from image {file_path}: {e}")
            return ExtractionResult(
                content=f"Error extracting text: {str(e)}",
                metadata={'error': str(e), 'file_path': file_path},
                success=False
            )
    
    def _get_ocr_config(self) -> str:
        """Build OCR configuration string"""
        config_parts = []
        
        if 'psm' in self.ocr_config:
            config_parts.append(f'--psm {self.ocr_config["psm"]}')
        if 'oem' in self.ocr_config:
            config_parts.append(f'--oem {self.ocr_config["oem"]}')
        
        default_configs = [
            '--dpi 300',
            '--oem 1',
            '--psm 3'
        ]
        
        config_parts.extend(default_configs)
        
        if self.ocr_config.get('preserve_interword_spaces'):
            config_parts.append('--preserve-interword-spaces 1')
        
        return ' '.join(config_parts)
    
    def _extract_metadata(self, image: 'Image.Image', file_path: str) -> Dict[str, Any]:
        """Extract metadata from image"""
        metadata = {
            'file_path': file_path,
            'file_size': os.path.getsize(file_path),
            'image_format': image.format,
            'image_mode': image.mode,
            'image_size': image.size,
            'ocr_languages': self.ocr_languages,
            'processing_time': datetime.now().isoformat()
        }
        
        try:
            if hasattr(image, '_getexif') and image._getexif():
                metadata['exif_data'] = dict(image._getexif())
        except:
            pass
        
        return metadata
    
    def extract_with_confidence(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text with confidence scores
        
        Args:
            file_path: Path to image file
            
        Returns:
            Dictionary with text, confidence scores, and bounding boxes
        """
        if not OCR_AVAILABLE:
            return {'error': 'OCR not available'}
        
        try:
            image = Image.open(file_path)
            
            data = pytesseract.image_to_data(
                image,
                lang='+'.join(self.ocr_languages),
                output_type=pytesseract.Output.DICT,
                config=self._get_ocr_config()
            )
            
            return {
                'text': ' '.join([word for word in data['text'] if word.strip()]),
                'confidence_scores': data['conf'],
                'bounding_boxes': list(zip(data['left'], data['top'], data['width'], data['height'])),
                'words': data['text'],
                'word_confidences': list(zip(data['text'], data['conf']))
            }
            
        except Exception as e:
            logger.error(f"Error in extract_with_confidence: {e}")
            return {'error': str(e)}
    
    def preprocess_image(self, image_path: str, output_path: Optional[str] = None) -> str:
        """
        Preprocess image for better OCR results
        
        Args:
            image_path: Input image path
            output_path: Optional output path
            
        Returns:
            Path to processed image
        """
        if not OCR_AVAILABLE:
            return image_path
        
        try:
            from PIL import ImageEnhance, ImageFilter
            
            image = Image.open(image_path)
            
            image = image.convert('L')
            
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)
            
            image = image.filter(ImageFilter.SHARPEN)
            
            if output_path:
                image.save(output_path)
                return output_path
            else:
                temp_path = Path(image_path).with_suffix('.processed.png')
                image.save(temp_path)
                return str(temp_path)
                
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image_path

def get_extractor_class():
    """Return extractor class for factory pattern"""
    return ImageExtractor
