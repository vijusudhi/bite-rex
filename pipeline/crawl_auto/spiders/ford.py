import scrapy

class ToScrapeCSSSpider(scrapy.Spider):
    name = "ford"

    start_urls = [
        'https://www.ford.com/',
    ]

    def parse(self, response):
        txt = response.css('p::text').getall()
        data = {}
        data['url'] = response.request.url
        data['text'] = txt
        yield data

        keys_to_consider = ['suvs', 
                            'powertrains',
                            'trucks',
                            'cars',
                            'performance',
                            'future'
                            ]

        next_page_url = response.css("a::attr(href)").getall()
        for next_url in next_page_url:
            if next_url is not None:
                for key in keys_to_consider:
                    if 'ford' in next_url and key in next_url:
                        yield scrapy.Request(response.urljoin(next_url))