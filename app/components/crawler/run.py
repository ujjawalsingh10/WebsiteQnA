from app.components.crawler.web_crawler import WebCrawler

# print('hlo')
crawler = WebCrawler('https://pmjay.gov.in')
crawler.crawl()