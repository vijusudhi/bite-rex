import scrapy

class ToScrapeCSSSpider(scrapy.Spider):
    name = "automotiveworld"

    start_urls = [
        'https://www.automotiveworld.com/magazine/',
    ]

    def parse(self, response):
        txt = response.css('p::text').getall()
        data = {}
        data['url'] = response.request.url
        data['text'] = txt
        yield data

        next_page_url = response.css("a::attr(href)").getall()
        for next_url in next_page_url:
            if next_url is not None:
                if 'automotiveworld' in next_url:
                    yield scrapy.Request(response.urljoin(next_url))