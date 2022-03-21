import scrapy

class ToScrapeCSSSpider(scrapy.Spider):
    name = "toyota"

    start_urls = [
        'https://www.toyota-europe.com/',
    ]

    def parse(self, response):
        txt = response.css('p::text').getall()
        data = {}
        data['url'] = response.request.url
        data['text'] = txt
        yield data

        keys_to_consider = ['world-of-toyota', 
                            'new-cars',
                            'service-and-accessories',
                            'business-customers',
                            ]

        next_page_url = response.css("a::attr(href)").getall()
        for next_url in next_page_url:
            if next_url is not None:
                for key in keys_to_consider:
                    if key in next_url:
                        yield scrapy.Request(response.urljoin(next_url))