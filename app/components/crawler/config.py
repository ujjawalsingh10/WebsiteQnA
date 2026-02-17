class CrawlerConfig:
    """
    Configuration related to website crawler
    """

    # Crawl Control
    MAX_DEPTH = 3
    MAX_PAGES = 500

    # Request behavior
    REQUEST_TIMEOUT = 10
    REQUEST_DELAY = 2
    MAX_RETRIES = 2

    # Header
    USER_AGENT = "GovChatbot/1.0 (CDAC, Noida Project - contact: iamujjawal10.us@gmail.com)"
    
    # File Storage
    BASE_STORAGE_PATH = 'data/raw'
    TEXT_FOLDER = 'pages'
    PDF_FOLDER = 'pdfs'
    IMAGE_FOLDER = 'images'

    # File types
    IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']
    