import scrapy

class ToScrapeCSSSpider(scrapy.Spider):
    name = "dummy"

    start_urls = [
        'https://www.gehalt.de/einkommen/suche/automotive+engineer',
    ]

    def parse(self, response):
        next_page_url = response.css("a::attr(href)").getall()
        for next_url in next_page_url:
            print(next_url)
        yield None