import hashlib
from urllib.parse import urljoin, urlparse, urlunparse

def normalize_url(base_url, link):
    """
    Convert relative to absolute url link by removing fragments like #team and remove trailing slash
    to stop infinite crawling 

    :param base_url: example - "https://pmjay.gov.in"
    :param link: example - "/about#team"

    """

    # make absolute url
    absolute_url = urljoin(base_url, link)
    parsed = urlparse(absolute_url)
    #remove fragment and query
    clean = parsed._replace(query="", fragment="")

    return urlunparse(clean).rstrip("/")

def is_internal_link(base_domain, url):
    """
    Ensure link belings to same domain
    """
    parsed_domain = urlparse(url)
    return parsed_domain.netloc == base_domain



# print(is_internal_link('pmjay.gov.in', 'https://pmjay.gov.in/terms-conditions'))