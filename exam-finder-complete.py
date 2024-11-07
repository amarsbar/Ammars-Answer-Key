import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
import sys
import os
import re
import io
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QTextEdit, QPushButton, QLabel, QScrollArea,
                           QHBoxLayout, QSpinBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import pymupdf
from rapidfuzz import fuzz
from PIL import Image, ImageOps

# DISCLAIMER: Code is severely bad! Probably has a bunch of methods I don't even use.

class MLQuestionMatcher:
    """ML-based question matching system"""
    def __init__(self):
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 3),
            stop_words='english',
            strip_accents='unicode',
            max_features=10000
        )
        self.questions = []
        self.metadata = []
        self.is_initialized = False

    def initialize_database(self, question_dir):
        """Initialize the question database from PDF files"""
        try:
            questions = []
            metadata = []
            
            for exam in os.listdir(question_dir):
                exam_path = os.path.join(question_dir, exam)
                doc = pymupdf.open(exam_path)
                
                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    blocks = page.get_text("blocks")
                    
                    for block in blocks:
                        block_text = block[4]
                        if len(block_text.strip()) > 20:  # Skip very short blocks
                            questions.append(block_text)
                            metadata.append({
                                'exam': exam,
                                'page_num': page_num,
                                'block': block,
                                'text': text
                            })
                
                doc.close()
            
            if questions:
                # Create embeddings
                self.question_embeddings = self.bert_model.encode(questions, convert_to_tensor=True)
                self.tfidf_matrix = self.tfidf.fit_transform(questions)
                self.questions = questions
                self.metadata = metadata
                self.is_initialized = True
                return True
            
            return False
            
        except Exception as e:
            print(f"Error initializing ML matcher: {str(e)}")
            return False

    def find_match(self, query_text, threshold=0.7):
        """Find the best matching question using ML techniques"""
        if not self.is_initialized or not query_text:
            return None
            
        try:
            # Get query embeddings
            query_embedding = self.bert_model.encode([query_text], convert_to_tensor=True)
            
            # Calculate semantic similarity
            with torch.no_grad():
                semantic_scores = cosine_similarity(
                    query_embedding.cpu().numpy(),
                    self.question_embeddings.cpu().numpy()
                )[0]
            
            # Get TF-IDF similarity
            query_tfidf = self.tfidf.transform([query_text])
            tfidf_scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
            
            # Get fuzzy match scores
            fuzzy_scores = np.array([
                fuzz.token_sort_ratio(query_text, q) / 100
                for q in self.questions
            ])
            
            # Combine scores with weights
            combined_scores = (
                0.4 * semantic_scores +
                0.3 * tfidf_scores +
                0.3 * fuzzy_scores
            )
            
            # Get best match
            best_idx = np.argmax(combined_scores)
            best_score = combined_scores[best_idx]
            
            if best_score >= threshold:
                metadata = self.metadata[best_idx]
                return {
                    'exam': metadata['exam'],
                    'text': metadata['text'],
                    'page': metadata['page_num'],
                    'ratio': float(best_score * 100),
                    'matched_text': metadata['block'][4]
                }
            
            return None
            
        except Exception as e:
            print(f"Error in ML matching: {str(e)}")
            return None

class ImageViewer(QScrollArea):
    """Custom image viewer with zoom functionality"""
    def __init__(self):
        super().__init__()
        self.zoom = 0
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.setWidget(self.image_label)
        self.setWidgetResizable(True)
        
        self.setStyleSheet("""
            QScrollArea {
                background-color: #000000;
                border: none;
            }
            QLabel {
                background-color: #000000;
            }
        """)
    
    def set_image(self, pixmap):
        self.original_pixmap = pixmap
        self.update_image()
    
    def update_image(self):
        if hasattr(self, 'original_pixmap'):
            scale = 1.0 + self.zoom / 10
            scaled_pixmap = self.original_pixmap.scaled(
                int(self.original_pixmap.width() * scale),
                int(self.original_pixmap.height() * scale),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
    
    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                self.zoom = min(50, self.zoom + 1)
            else:
                self.zoom = max(-5, self.zoom - 1)
            self.update_image()
        else:
            super().wheelEvent(event)

class ModernTextEdit(QTextEdit):
    """Styled text edit for question input"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d2d;
                color: #ffffff;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 14px;
                line-height: 1.4;
            }
            QTextEdit::placeholder {
                color: #888888;
            }
        """)

class ExamFinderApp(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        
        # Set up directory paths
        if getattr(sys, 'frozen', False):
            self.app_dir = os.path.dirname(sys.executable)
        else:
            self.app_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.question_dir = os.path.join(self.app_dir, "Physics Exam", "Questions")
        self.answer_dir = os.path.join(self.app_dir, "Physics Exam", "AnswerKeys")
        
        # Create directories if they don't exist
        os.makedirs(self.question_dir, exist_ok=True)
        os.makedirs(self.answer_dir, exist_ok=True)

        # MACHINE LEARNING YUH YUH (Initialize ML matcher)
        self.ml_matcher = MLQuestionMatcher()
        self.initialize_ml_matcher()
    
    def initialize_ml_matcher(self):
        """Initialize the ML-based question matcher"""
        try:
            success = self.ml_matcher.initialize_database(self.question_dir)
            if success:
                print("ML matcher initialized successfully")
            else:
                print("Failed to initialize ML matcher")
        except Exception as e:
            print(f"Error initializing ML matcher: {str(e)}")

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Exam Question Finder")
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QPushButton {
                background-color: #0d47a1;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
                color: white;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QSpinBox {
                background-color: #2d2d2d;
                color: #ffffff;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
                min-width: 80px;
            }
            QLabel {
                color: #888888;
                font-size: 12px;
            }
        """)
        
        # Main layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # Top section with question input and source
        top_section = QWidget()
        top_layout = QVBoxLayout(top_section)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(5)
        
        # Question source label
        self.source_label = QLabel()
        self.source_label.setStyleSheet("""
            QLabel {
                color: #4caf50;
                font-size: 12px;
                padding: 2px;
            }
        """)
        top_layout.addWidget(self.source_label)
        
        # Question text input with number
        question_row = QWidget()
        question_layout = QHBoxLayout(question_row)
        question_layout.setContentsMargins(0, 0, 0, 0)
        question_layout.setSpacing(10)
        
        # Question input on the left
        self.question_input = ModernTextEdit()
        self.question_input.setPlaceholderText("Enter question text...")
        self.question_input.setMaximumHeight(100)
        question_layout.addWidget(self.question_input, stretch=4)
        
        # Question number on the right
        number_widget = QWidget()
        number_layout = QVBoxLayout(number_widget)
        number_layout.setContentsMargins(0, 0, 0, 0)
        number_layout.setSpacing(5)
        
        number_label = QLabel("Question Number:")
        number_label.setStyleSheet("color: #ffffff; font-size: 14px;")
        self.number_input = QSpinBox()
        self.number_input.setSpecialValueText("Auto")
        self.number_input.setRange(0, 99)
        self.number_input.setValue(0)
        
        number_layout.addWidget(number_label)
        number_layout.addWidget(self.number_input)
        number_layout.addStretch()
        
        question_layout.addWidget(number_widget, stretch=1)
        top_layout.addWidget(question_row)
        
        # Search button
        search_button = QPushButton("Find Question")
        search_button.clicked.connect(self.find_question)
        top_layout.addWidget(search_button)
        
        layout.addWidget(top_section)
        
        # Answer label
        self.answer_label = QLabel()
        self.answer_label.setStyleSheet("""
            QLabel {
                color: #4caf50;
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
                background-color: #2d2d2d;
                border-radius: 5px;
                margin: 5px 0;
            }
        """)
        layout.addWidget(self.answer_label)
        
        # Image viewer
        self.image_viewer = ImageViewer()
        self.image_viewer.setMinimumHeight(400)
        layout.addWidget(self.image_viewer, 1)
        
        # Status label
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #888888; padding: 5px;")
        layout.addWidget(self.status_label)
        
        self.setMinimumSize(800, 800)

    def find_fuzzy_match(self, question_text):
        """Find fuzzy matches using character-level matching"""
        best_match = None
        best_ratio = 65
        highest_ratio = 0
        
        # Normalize input text
        question_text = question_text.lower().strip()
        
        # Create overlapping segments
        segments = []
        segment_length = min(50, len(question_text))
        overlap = 25
        
        for i in range(0, len(question_text) - segment_length + 1, overlap):
            segment = question_text[i:i + segment_length]
            if len(segment) >= 20:
                segments.append(segment)
        
        if not segments:
            segments = [question_text]
        
        for exam in os.listdir(self.question_dir):
            exam_path = os.path.join(self.question_dir, exam)
            try:
                doc = pymupdf.open(exam_path)
                for page in doc:
                    text = page.get_text()
                    text_lower = text.lower()
                    
                    # Use sliding window
                    window_size = segment_length + 20
                    
                    for i in range(0, len(text_lower) - window_size + 1, 10):
                        window = text_lower[i:i + window_size]
                        
                        for segment in segments:
                            # Calculate multiple similarity metrics
                            char_ratio = fuzz.ratio(segment, window)
                            partial_ratio = fuzz.partial_ratio(segment, window)
                            token_sort = fuzz.token_sort_ratio(segment, window)
                            token_set = fuzz.token_set_ratio(segment, window)
                            
                            max_ratio = max(char_ratio, 
                                          partial_ratio * 0.95,
                                          token_sort * 0.9,
                                          token_set * 0.85)
                            
                            # Word overlap bonus
                            segment_words = set(segment.split())
                            window_words = set(window.split())
                            word_overlap = len(segment_words.intersection(window_words)) / len(segment_words)
                            
                            if word_overlap > 0.6:
                                max_ratio *= (1 + word_overlap * 0.2)
                            
                            if max_ratio > best_ratio and max_ratio > highest_ratio:
                                # Get context around match
                                start_idx = max(0, i - 100)
                                end_idx = min(len(text), i + window_size + 100)
                                matched_text = text[start_idx:end_idx].strip()
                                
                                highest_ratio = max_ratio
                                best_match = {
                                    'exam': exam,
                                    'text': text,
                                    'page': page,
                                    'ratio': max_ratio,
                                    'matched_text': matched_text
                                }
                
                doc.close()
                
            except Exception as e:
                print(f"Error processing {exam}: {str(e)}")
                if 'doc' in locals():
                    doc.close()
        
        return best_match

    def find_exact_match(self, question_text):
        """Find exact matches in the exam PDFs"""
        for exam in os.listdir(self.question_dir):
            exam_path = os.path.join(self.question_dir, exam)
            try:
                doc = pymupdf.open(exam_path)
                for page in doc:
                    text = page.get_text()
                    if question_text in text:
                        match = {
                            'exam': exam,
                            'text': text,
                            'page': page,
                            'ratio': 100,
                            'matched_text': question_text
                        }
                        doc.close()
                        return match
                doc.close()
            except Exception as e:
                print(f"Error processing {exam}: {str(e)}")
        return None

    def search_for_question(self, original_question):
        """Enhanced search using both ML and traditional methods"""
        try:
            # Try ML-based matching first
            if self.ml_matcher.is_initialized:
                ml_match = self.ml_matcher.find_match(original_question)
                if ml_match and ml_match['ratio'] > 75:
                    return ml_match
            
            # Fall back to traditional matching methods
            normalized_question = original_question.strip()
            
            # Try exact match
            exact_match = self.find_exact_match(normalized_question)
            if exact_match:
                return exact_match
            
            # Try case-insensitive match
            case_insensitive_match = self.find_exact_match(normalized_question.lower())
            if case_insensitive_match:
                return case_insensitive_match
            
            # Try fuzzy match (this thing never works tbh)
            fuzzy_match = self.find_fuzzy_match(normalized_question)
            if fuzzy_match and fuzzy_match['ratio'] > 70:
                return fuzzy_match
            
            # Try with shortened question
            if len(normalized_question.split()) > 5:
                key_words = ' '.join(normalized_question.split()[:5])
                shortened_match = self.find_fuzzy_match(key_words)
                if shortened_match and shortened_match['ratio'] > 75:
                    return shortened_match
            
            return None
            
        except Exception as e:
            print(f"Error in search: {str(e)}")
            return None

    def extract_question_image(self, page, question_text):
        """Extract question area and surrounding context from PDF"""
        blocks = page.get_text("blocks")
        
        question_block = None
        for block in blocks:
            if question_text in block[4]:
                question_block = block
                break
        
        if question_block:
            x0, y0, x1, y1 = question_block[:4]
            padding = 50
            extra_height = 400
            extra_space_above = 200
            
            clip_rect = pymupdf.Rect(
                max(0, x0 - padding),
                max(0, y0 - extra_space_above),
                x1 + padding,
                y1 + extra_height
            )
            matrix = pymupdf.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=matrix, clip=clip_rect)
            return pix
        
        return None

    def display_question(self, match):
        """Helper method to display the question image"""
        try:
            doc = pymupdf.open(os.path.join(self.question_dir, match['exam']))
            pages = list(doc)
            for page in pages:
                if match['matched_text'] in page.get_text():
                    pixmap = self.extract_question_image(page, match['matched_text'])
                    if pixmap:
                        self.display_image(pixmap)
                    break
            doc.close()
        except Exception as e:
            self.status_label.setText(f"Error displaying question: {str(e)}")

    def display_image(self, pixmap):
        """Convert and display the image"""
        try:
            img_data = pixmap.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_data))
            inverted_image = ImageOps.invert(pil_image.convert('RGB'))
            
            img_byte_arr = io.BytesIO()
            inverted_image.save(img_byte_arr, format='PNG', quality=95)
            img_byte_arr = img_byte_arr.getvalue()
            
            q_image = QImage.fromData(img_byte_arr)
            pixmap = QPixmap.fromImage(q_image)
            self.image_viewer.set_image(pixmap)
            
        except Exception as e:
            self.status_label.setText(f"Error displaying image: {str(e)}")

    def find_answer(self, exam_name, question_number):
        """Find the answer in the answer key"""
        try:
            answer_file = exam_name.replace("Exam.pdf", "Key.pdf")
            answer_path = os.path.join(self.answer_dir, answer_file)
            
            if not os.path.exists(answer_path):
                self.status_label.setText(f"Answer key not found: {answer_file}")
                return None
            
            doc = pymupdf.open(answer_path)
            for page in doc:
                text = page.get_text()
                
                # Try newer format (number. A)
                pattern = fr'{question_number}\.\s+([A-D])\s+'
                match = re.search(pattern, text)
                if match:
                    answer = match.group(1)
                    doc.close()
                    return answer
                
                # Try older format (5. H 1 B 2)
                pattern = fr'{question_number}\.\s+[A-Z]\s+\d+\s+([A-D])\s+'
                match = re.search(pattern, text)
                if match:
                    answer = match.group(1)
                    doc.close()
                    return answer
            
            doc.close()
            return None
            
        except Exception as e:
            self.status_label.setText(f"Error finding answer: {str(e)}")
            return None

    def find_question_number(self, text, question_text):
        """Find the question number from the text"""
        question_index = text.find(question_text)
        if question_index == -1:
            return None
        
        preceding_text = text[:question_index]
        matches = re.finditer(r'(\d+)[\.\)\s]', preceding_text)
        matches_list = list(matches)
        
        if matches_list:
            return int(matches_list[-1].group(1))
        return None
    
    def format_exam_name(self, exam_name):
        """Format exam name with proper year handling"""
        exam_match = re.match(r'([A-Za-z]+)(\d{2})Exam\.pdf', exam_name)
        if exam_match:
            month, year = exam_match.groups()
            year_num = int(year)
            full_year = 1900 + year_num if year_num > 50 else 2000 + year_num
            return f"{month} {full_year} Exam"
        return exam_name.replace(".pdf", "")

    def find_question(self):
        """Main function to find and display question information"""
        try:
            question_text = self.question_input.toPlainText().strip()
            if not question_text:
                self.status_label.setText("Please enter question text")
                return

            # Search for the question
            match = self.search_for_question(question_text)
            
            if match:
                try:
                    # Update source label immediately when match is found
                    formatted_exam = self.format_exam_name(match['exam'])
                    self.source_label.setText(f"From: {formatted_exam}")
                    
                    # Get question number (auto or manual)
                    question_num = self.number_input.value()
                    if question_num == 0:  # Auto mode
                        question_num = self.find_question_number(match['text'], match['matched_text'])
                        if question_num:
                            self.number_input.setValue(question_num)
                        else:
                            self.status_label.setText("Could not auto-detect question number. Please specify manually.")
                            self.display_question(match)
                            return
                    
                    # Find answer using question number
                    answer = self.find_answer(match['exam'], question_num)
                    
                    if answer:
                        self.answer_label.setText(f"Answer: {answer}")
                        self.display_question(match)
                        
                        if match['ratio'] < 100:
                            self.status_label.setText(
                                f"Showing match with {int(match['ratio'])}% confidence. Please verify."
                            )
                        else:
                            self.status_label.clear()
                    else:
                        self.status_label.setText("Could not find answer in answer key")
                        self.display_question(match)
                except Exception as e:
                    self.status_label.setText(f"Error processing match: {str(e)}")
            else:
                self.status_label.setText("No matching question found")
                    
        except Exception as e:
            self.status_label.setText(f"An error occurred: {str(e)}")

def main():
    """Main entry point of the application"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = ExamFinderApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()