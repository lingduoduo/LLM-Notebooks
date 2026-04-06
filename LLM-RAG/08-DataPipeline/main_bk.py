import os
from ocr_module import OCREngine
from processor_module import TextProcessor
from dify_module import DifyUploader
from metadata_module import bind_metadata

def main(image_path):
    # OCR提取文本
    ocr_engine = OCREngine()
    raw_text = ocr_engine.extract_text(image_path)
    print(f"--debug--: OCR \n {raw_text}")
    
    # 文本预处理与分段
    processor = TextProcessor()
    segments = processor.process(raw_text)
    print(f"--debug--: 文本预处理与分段 \n {segments}")
    
    # 上传至Dify
    uploader = DifyUploader()
    document_id = uploader.upload_document_by_text(segments)
    print(f"--debug--: Dify 文档ID \n {document_id}")
    
    # 绑定元数据
    metadata = [
        {'name': 'author', 'value': 'default_author'},
        {'name': 'doc_source', 'value': os.path.basename(image_path)}
    ]
    bind_metadata(document_id, metadata)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(base_dir, "example.png")
    main(image_path)