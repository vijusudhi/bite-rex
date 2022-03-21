import scrapy

class ToScrapeCSSSpider(scrapy.Spider):
    name = "bosch"
    # allowed_domains = [
    #     'https://www.bosch-mobility-solutions.com/'
    # ]
    start_urls = [
        'https://www.bosch-mobility-solutions.com/en/',
    ]

    def parse(self, response):
        txt = response.css('p::text').getall()
        data = {}
        data['url'] = response.request.url
        data['text'] = txt
        yield data

        next_page_url = response.css("a::attr(href)").getall()
        for next_url in next_page_url:
            # print('Analysing url', next_url)
            if next_url is not None:
                if 'solutions' in next_url or 'mobility-topics' in next_url:
                    yield scrapy.Request(response.urljoin(next_url))


