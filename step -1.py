import os
import nltk
from PyPDF2 import PdfReader, PdfWriter
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTFigure
from PIL import Image
from pdf2image import convert_from_path
import pytesseract
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

pytesseract.pytesseract.tesseract_cmd = r"G:\Tesseract\tesseract.exe"

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def text_extraction(element):
    line_text = element.get_text()
    line_formats = []
    for text_line in element:
        if isinstance(text_line, LTTextContainer):
            for character in text_line:
                if isinstance(character, LTChar):
                    line_formats.append(character.fontname)
                    line_formats.append(character.size)
    format_per_line = list(set(line_formats))
    return (line_text, format_per_line)


def convert_to_images(input_file):
    images = convert_from_path(input_file)
    image = images[0]
    output_file = 'PDF_image.png'
    image.save(output_file, 'PNG')


def image_to_text(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text


def crop_image(element, pageObj):
    [image_left, image_top, image_right, image_bottom] = [
        element.x0, element.y0, element.x1, element.y1]

    cropped_pdf_writer = PdfWriter()
    cropped_page = pageObj
    cropped_page.mediabox.lower_left = (image_left, image_bottom)
    cropped_page.mediabox.upper_right = (image_right, image_top)

    cropped_pdf_writer.add_page(cropped_page)

    with open('cropped_image.pdf', 'wb') as cropped_pdf_file:
        cropped_pdf_writer.write(cropped_pdf_file)


def extract_text_from_pdf(pdf_path):
    pdfFileObj = open(pdf_path, 'rb')
    pdfReaded = PdfReader(pdfFileObj)
    text_per_page = {}
    image_flag = False

    for pagenum, page in enumerate(extract_pages(pdf_path)):
        pageObj = pdfReaded.pages[pagenum]
        page_text = []
        line_format = []
        text_from_images = []
        page_content = []

        page_elements = [(element.y1, element) for element in page._objs]
        page_elements.sort(key=lambda a: a[0], reverse=True)

        for i, component in enumerate(page_elements):
            element = component[1]
            if isinstance(element, LTTextContainer):
                (line_text, format_per_line) = text_extraction(element)
                page_text.append(line_text)
                line_format.append(format_per_line)
                page_content.append(line_text)
            if isinstance(element, LTFigure):
                crop_image(element, pageObj)
                convert_to_images('cropped_image.pdf')
                image_text = image_to_text('PDF_image.png')
                text_from_images.append(image_text)
                page_content.append(image_text)
                page_text.append('image')
                line_format.append('image')
                image_flag = True

        dctkey = 'Page_'+str(pagenum)
        text_per_page[dctkey] = [page_text,
                                 line_format, text_from_images, page_content]

    pdfFileObj.close()

    if image_flag:
        os.remove('cropped_image.pdf')
        os.remove('PDF_image.png')

    result = ''.join(text_per_page['Page_0'][3])
    return result


def read_resumes_from_folder(folder_path):
    resumes = []
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return resumes

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            print(f"Reading file: {file_path}")
            try:
                extracted_text = extract_text_from_pdf(file_path)
                if extracted_text:
                    resumes.append(extracted_text)
                else:
                    print(
                        f"File {file_path} is empty or could not extract text!")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return resumes


def preprocess_resumes(resumes):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    preprocessed_resumes = []

    for resume in resumes:
        print("Original Resume Text:\n", resume)

        resume = resume.lower()
        tokens = word_tokenize(resume)

        print("Tokens after tokenization:\n", tokens)

        tokens = [word for word in tokens if word not in stop_words]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        if tokens:
            preprocessed_resumes.append(' '.join(tokens))
        else:
            print(
                "All tokens removed after stop words or lemmatization, skipping this resume.")

    return preprocessed_resumes


def vectorize_resumes(preprocessed_resumes):
    if not preprocessed_resumes:
        raise ValueError("No valid resumes left after preprocessing.")

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_resumes)

    print("Final TF-IDF Matrix:\n", tfidf_matrix.toarray())
    print("Shape of the TF-IDF Matrix:", tfidf_matrix.shape)

    return tfidf_matrix


def process_resumes_folder(folder_path):
    resumes = read_resumes_from_folder(folder_path)
    print(f"Read {len(resumes)} resumes from folder '{folder_path}'.")

    if not resumes:
        print("No resumes found. Exiting.")
        return

    preprocessed_resumes = preprocess_resumes(resumes)
    print("Preprocessing completed.")

    tfidf_matrix = vectorize_resumes(preprocessed_resumes)
    print(
        f"TF-IDF vectorization completed. Matrix shape: {tfidf_matrix.shape}")

    return tfidf_matrix


folder_path = r"G:\Resume Algorithm\Resume"
tfidf_matrix = process_resumes_folder(folder_path)

if tfidf_matrix is not None:
    print("TF-IDF matrix generated from all resumes:", tfidf_matrix.shape)
