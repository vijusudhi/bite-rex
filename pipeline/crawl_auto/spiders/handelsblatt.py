import scrapy

class ToScrapeCSSSpider(scrapy.Spider):
    name = "handelsblatt"

    start_urls = [
        'https://www.handelsblatt.com/themen/automobilindustrie',
    ]

    def parse(self, response):
        txt = response.css('p::text').getall()
        data = {}
        data['url'] = response.request.url
        data['text'] = txt
        yield data

        keys_to_consider = ['industrie', 
                            'mobilitaet',
                            'technik',
                            'bmw',
                            'audi',
                            'volkswagen',
                            'mercedes-benz'
                            ]

        next_page_url = response.css("a::attr(href)").getall()
        for next_url in next_page_url:
            if next_url is not None:
                for key in keys_to_consider:
                    if key in next_url:
                        yield scrapy.Request(response.urljoin(next_url))