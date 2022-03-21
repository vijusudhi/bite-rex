import scrapy

class ToScrapeCSSSpider(scrapy.Spider):
    name = "bmwgroup_en"

    start_urls = [
        'https://www.bmwgroup.com/en.html',
    ]

    def parse(self, response):
        txt = response.css('p::text').getall()
        data = {}
        data['url'] = response.request.url
        data['text'] = txt
        yield data

        keys_to_consider = ['company', 
                            'responsibility',
                            'innovation',
                            'brands-and-services',
                            'electromobility'
                            'news'
                            ]

        next_page_url = response.css("a::attr(href)").getall()
        for next_url in next_page_url:
            if next_url is not None:
                for key in keys_to_consider:
                    if key in next_url:
                        yield scrapy.Request(response.urljoin(next_url))
