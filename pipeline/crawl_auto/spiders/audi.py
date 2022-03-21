import scrapy

class ToScrapeCSSSpider(scrapy.Spider):
    name = "audi"
    # allowed_domains = [
    #     'https://www.bosch-mobility-solutions.com/'
    # ]
    start_urls = [
        'https://www.audi-technology-portal.de/en/',
    ]

    def parse(self, response):
        txt = response.css('p::text').getall()
        data = {}
        data['url'] = response.request.url
        data['text'] = txt
        yield data

        next_page_url = response.css("a::attr(href)").getall()
        # print(next_page_url)
        for next_url in next_page_url:
            if next_url is not None:
                if 'audi' in next_url:
                    yield scrapy.Request(response.urljoin(next_url))


